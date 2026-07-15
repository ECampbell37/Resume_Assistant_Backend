# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from analysis import analyze_resume
from chatbot import extract_text_from_pdf, get_or_create_chatbot
from job_match import run_job_match
from revision import rewrite_resume
from fastapi.responses import PlainTextResponse
from typing import List, Optional
from pydantic import BaseModel
from resume_builder import build_resume

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://resume-assistant.website",
        "https://www.resume-assistant.website",
        "https://resume-assistant-inky.vercel.app",
        "http://localhost:3000",
        "https://resume-assistant-backend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory per-user data
user_memory = {}  # { user_id: file_bytes }


############ Health ###################3

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}


############ Analysis #################

@app.post("/analyze")
async def analyze(user_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    file_bytes = await file.read()
    
    # File size check
    if len(file_bytes) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Resume file is too large. Max allowed size is 2MB.")
    
    user_memory[user_id] = file_bytes  # Store resume for chatbot

    try:
        result = analyze_resume(file_bytes)  # might raise page len error
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")



############ Chatbot #################

@app.post("/chatbot/load")
async def load_resume_for_chatbot(user_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")
    file_bytes = await file.read()
    
    # File size check
    if len(file_bytes) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Resume file is too large. Max allowed size is 2MB.")
    
    user_memory[user_id] = file_bytes
    return {"status": "ok", "message": "Resume loaded into chatbot memory."}

@app.post("/chatbot/respond")
async def resume_chat(user_id: str = Form(...), message: str = Form(...)):
    if user_id not in user_memory:
        raise HTTPException(status_code=404, detail="No resume found in chatbot memory. Please reload it.")
    try:
        resume_bytes = user_memory[user_id]
        resume_text = extract_text_from_pdf(resume_bytes)  # might raise page length error
        chain = get_or_create_chatbot(user_id, resume_text)
        response = chain.run(message)
        return JSONResponse(content={"response": response})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

    
    
    
############ Job Match #################

@app.post("/jobmatch")
async def job_match(user_id: str = Form(...), job_description: str = Form(...)):
    if user_id not in user_memory:
        raise HTTPException(status_code=404, detail="Resume not found. Please upload first.")

    try:
        resume_bytes = user_memory[user_id]
        resume_text = extract_text_from_pdf(resume_bytes)  # might raise page len error
        result = run_job_match(resume_text, job_description)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Job match failed: {str(e)}")



############ Revision Mode #################


@app.post("/revision")
async def revision_mode(user_id: str = Form(...)):
    if user_id not in user_memory:
        raise HTTPException(status_code=404, detail="No resume found. Please load it first.")

    try:
        file_bytes = user_memory[user_id]
        resume_text = extract_text_from_pdf(file_bytes)
        rewritten_text = rewrite_resume(resume_text)
        return PlainTextResponse(content=rewritten_text)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume rewrite failed: {str(e)}")





# --- Pydantic models for the create-resume form ---

class PersonalInfoModel(BaseModel):
    fullName: str
    email: Optional[str] = ""
    phone: Optional[str] = ""
    location: Optional[str] = ""
    linkedin: Optional[str] = ""
    otherLinks: Optional[str] = ""

class ExperienceModel(BaseModel):
    jobTitle: str
    company: str
    location: Optional[str] = ""
    startDate: Optional[str] = ""
    endDate: Optional[str] = ""
    responsibilities: Optional[str] = ""

class EducationModel(BaseModel):
    school: str
    degree: Optional[str] = ""
    fieldOfStudy: Optional[str] = ""
    location: Optional[str] = ""
    graduationDate: Optional[str] = ""
    honors: Optional[str] = ""

class ProjectModel(BaseModel):
    name: str
    description: Optional[str] = ""
    technologies: Optional[str] = ""

class ResumeFormData(BaseModel):
    personalInfo: PersonalInfoModel
    summary: Optional[str] = ""
    experience: List[ExperienceModel] = []
    education: List[EducationModel] = []
    skills: Optional[str] = ""
    projects: List[ProjectModel] = []
    certifications: Optional[str] = ""
    

############ Create Resume #################

@app.post("/generate-resume")
async def generate_resume(data: ResumeFormData):
    if not data.personalInfo.fullName.strip():
        raise HTTPException(status_code=400, detail="Full name is required.")
    if not data.personalInfo.email.strip():
        raise HTTPException(status_code=400, detail="Email is required.")
    if not data.personalInfo.location.strip():
        raise HTTPException(status_code=400, detail="Location is required.")
    if not data.experience or any(not e.jobTitle.strip() or not e.company.strip() for e in data.experience):
        raise HTTPException(status_code=400, detail="At least one complete job entry is required.")
    if not data.education or any(not e.school.strip() for e in data.education):
        raise HTTPException(status_code=400, detail="At least one complete education entry is required.")

    try:
        contact_parts = [
            v.strip()
            for v in [
                data.personalInfo.email,
                data.personalInfo.phone,
                data.personalInfo.location,
                data.personalInfo.linkedin,
                data.personalInfo.otherLinks,
            ]
            if v and v.strip()
        ]
        contact_line = " | ".join(contact_parts)

        body_markdown = build_resume(data.dict())

        header = f"# {data.personalInfo.fullName.strip()}"
        if contact_line:
            header += f"\n\n{contact_line}"

        full_resume = f"{header}\n\n{body_markdown}".strip()
        return PlainTextResponse(content=full_resume)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume generation failed: {str(e)}")