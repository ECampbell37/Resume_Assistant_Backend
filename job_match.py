#job_match.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import json
import re

llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

job_match_prompt = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="""
You are an expert career advisor AI. Evaluate how well the resume below matches the job description provided.
Base your evaluation ONLY on the text content, not formatting.
Try to be as hyperspecific and as personalized to the resume and candidate as possible.

Write the response **directly to the applicant** using “you” (e.g., “you should apply”) — never use their name or refer to them in the third person.

---- RESUME ----
{resume}

---- JOB DESCRIPTION ----
{job_description}


Evaluate the match according to the following fit levels:

- "Underqualified" → Lacks most of the core skills or experience.
- "Somewhat Qualified" → Meets a few requirements but is missing many core skills or relevant experience.
- "Good Fit" → Meets many requirements with decent evidence of relevant skills or background. However, something important is holding the candidate back from being stronger.
- "Strong Match" → Meets most major requirements with solid alignment in both skills and experience.
- "Ideal Match" → The candidate is exactly the type of person that the hiring team is looking for. The candidate would be a highly competitive applicant.

Do not underrate. For example, if the candidate is a strong, competitive applicant for the specified position, do not give them "Good Fit". Give them Strong or Ideal Match.
Always round up, and don't be afraid to give a rating of "Ideal Match" if the candidate deserves it. We want them to be confident and succeed! You should want to give the candidate a rating of ideal.
The candidate does not need to match every skill in the job description to earn a rating of ideal. A highly competitive applicant should always earn "Ideal Match" if the role is meant for them. 
Take an all encompassing view, including where the candidate is in their career, the type of company it is, what the target audience for the job post is, and what the hiring team's relative expectations are. Take a deep, nauanced look at the match. 
If the candidate is missing multiple years of required experience, they cannot earn higher than "Somewhat Qualified. This is an impotant factor.
If the candidate is missing a large number of required skills or expectations for the role, they cannot earn higher than "Good Fit".
Don't be too picky about industry knowledge. If the candidate is highly competitive, but lacks specific industry knowledge or industry experience, you should still give them "Ideal Match". 

Provide a JSON response with the following fields:
- fit_category: One of the values above. Make sure it matches your analysis and recommendation.
- matched_skills: List of resume elements and specific skills that align with job requirements.
- missing_skills: List of important skills or experiences from the job description that the resume is missing. Even if the candidate is ideal, list whatever expectations they do not match with the job description.
- recommendation: A charismatic, concise, straight to the point paragraph written directly to the applicant. Include your recommendation to apply or not, and explain your rationale in a way that clearly supports the fit_category you've chosen. If saying they should apply, be encouraging. If they are missing years of experience, be realistic.

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
