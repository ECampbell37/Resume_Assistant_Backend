# main.py

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from analysis import analyze_resume

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary per-user memory (not persistent)
user_memory = {}  # { user_id: file_bytes }

@app.post("/analyze")
async def analyze(user_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    file_bytes = await file.read()
    user_memory[user_id] = file_bytes

    try:
        result = analyze_resume(file_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
