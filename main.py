# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from analysis import analyze_resume
from chatbot import extract_text_from_pdf, get_or_create_chatbot
from job_match import run_job_match

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory per-user data
user_memory = {}  # { user_id: file_bytes }


############ Analysis #################

@app.post("/analyze")
async def analyze(user_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    file_bytes = await file.read()
    user_memory[user_id] = file_bytes  # Store resume for chatbot

    try:
        result = analyze_resume(file_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


############ Chatbot #################

@app.post("/chatbot/load")
async def load_resume_for_chatbot(user_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")
    file_bytes = await file.read()
    user_memory[user_id] = file_bytes
    return {"status": "ok", "message": "Resume loaded into chatbot memory."}

@app.post("/chatbot/respond")
async def resume_chat(user_id: str = Form(...), message: str = Form(...)):
    if user_id not in user_memory:
        raise HTTPException(status_code=404, detail="No resume found in chatbot memory. Please reload it.")
    try:
        resume_bytes = user_memory[user_id]
        resume_text = extract_text_from_pdf(resume_bytes)
        chain = get_or_create_chatbot(user_id, resume_text)
        response = chain.run(message)
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")
    
    
    
############ Job Match #################

@app.post("/jobmatch")
async def job_match(user_id: str = Form(...), job_description: str = Form(...)):
    if user_id not in user_memory:
        raise HTTPException(status_code=404, detail="Resume not found. Please upload first.")

    try:
        resume_bytes = user_memory[user_id]
        resume_text = extract_text_from_pdf(resume_bytes)
        result = run_job_match(resume_text, job_description)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Job match failed: {str(e)}")
