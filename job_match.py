#job_match.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import json
import re

llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

job_match_prompt = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="""
You are an expert career advisor AI. Evaluate how well the resume below matches the job description provided.
Base your evaluation ONLY on the text content, not formatting.

Write the response **directly to the applicant** using “you” (e.g., “you should apply”) — never use their name or refer to them in the third person.

---- RESUME ----
{resume}

---- JOB DESCRIPTION ----
{job_description}

Provide a JSON response with the following fields:
- match_percentage: A number from 0 to 100 representing how well the resume fits the job.
- fit_category: One of "Underqualified", "Slightly Underqualified", "Ideal", "Slightly Overqualified", or "Overqualified".
- matched_skills: List of resume elements that align with job requirements.
- missing_skills: List of important skills/requirements from the job that the resume is missing.
- recommendation: One paragraph with a professional recommendation about whether the person should apply, and why.

Respond only in raw JSON format.
"""
)

job_match_chain = LLMChain(llm=llm, prompt=job_match_prompt)

def run_job_match(resume_text: str, job_description: str) -> dict:
    raw_output = job_match_chain.run(resume=resume_text, job_description=job_description)

    # ✅ Strip Markdown code fences if present
    cleaned = re.sub(r"^```json|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Failed to parse JSON:\n\n{cleaned}\n\nError: {e}")
