# analysis.py

import os
import warnings

import fitz  # PyMuPDF

# Langchain
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")


# Load API key from env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the GPT Model
llm_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.7, model=llm_model)


########### Prompts & chains #####################

# Summarization
summary_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    You are a professional, friendly, and detail-oriented career assistant. \
    Speak directly to the candidate using “you.” Be specific, constructive, and easy to understand. \
    Summarize the resume below in 3–5 concise sentences. \
    Focus on the candidate's background, relevant experience, technical and soft skills, and any notable achievements. \
    Speak directly to the candidate using 'you' (e.g., 'You graduated with...'). \
    Avoid generic filler — highlight what makes them stand out. \
    This summary is meant to give the candidate an idea of what a recruiter would see in their resume. \
    \n\n{resume}
    """
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# Rating
rating_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    You are a professional, friendly, and detail-oriented career assistant. \
    Be specific, constructive, and easy to understand. \

    Rate the following resume on a scale of 0–100% based on the rubric below. You are speaking *to* the \
    candidate, so refer to them as "you" (not by name).

    Focus only on the *text content* — ignore visual formatting or layout.

    ---

    ### Resume Evaluation Rubric (100 pts total)

    1. **Personal Information (20 pts)**
    - Full name is included (5 pts)
    - At least two contact methods (email, phone, LinkedIn, etc.) are clearly provided (5 pts)
    - Location (city, state, or general region) is present (3 pts)
    - Any links (LinkedIn, portfolio, GitHub) are written out in full (5 pts)
    - Optional: a role headline or title like “Data Analyst” (2 pts)

    2. **Summary or Objective (10 pts)**
    - Clear professional identity or career goal (3 pts)
    - Highlights strengths, experience, or goals (4 pts)
    - Professional and concise tone (3 pts)

    3. **Work Experience (20 pts)**
    - Job titles, companies, and dates are stated (5 pts)
    - Responsibilities and/or achievements are described (5 pts)
    - Specific or measurable examples are used (5 pts)
    - Career relevance or growth is shown (5 pts)

    4. **Education (20 pts)**
    - School name and degree/certification listed (5 pts)
    - Graduation date or timeframe included (3 pts)
    - Honors, awards, or academic highlights mentioned (4 pts)
    - Educational background is relevant or explained (5 pts)
    - Optional: relevant courses or training listed (3 pts)

    5. **Skills and/or Projects (20 pts)**
    - Relevant hard skills (tools, software, languages) listed (6 pts)
    - Soft skills or workplace competencies included (3 pts)
    - Projects (if present) include title, context, tools used (6 pts)
    - Projects show initiative or relevance (5 pts)

    6. **Writing Quality (10 pts)**
    - Spelling and grammar are strong (4 pts)
    - Professional, concise word choice (3 pts)
    - Clear structure and logical section flow (3 pts)

    ---

    First, write a line with the score:  
    **Overall Rating: __%**

    Then, on the next line, write a paragraph explaining your reasoning. Highlight strengths, weaknesses, 
    and suggest specific areas to improve.

    Return individual scores for each section in parentheses, like:  
    Rating: 85% (Personal Info: 18/20, Summary: 9/10, Work Exp: 16/20, Education: 17/20, Skills/Projects: 17/20, Writing: 8/10)


    Do not suggest visual layout or formatting changes
    
    Here is the resume to evaluate:

    {resume}
    """
)
rating_chain = LLMChain(llm=llm, prompt=rating_prompt, output_key="rating")

# Personal Info Extraction
info_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    Extract the following information from the resume: Full Name, Email, Phone Number, Location (city/state), \
    and relevant links (LinkedIn, GitHub, Portfolio, etc.).  
    Use this format (one per line):  
    Name: ___  
    Email: ___  
    Phone Number: ___  
    Location: ___  
    LinkedIn: ___  
    Other Links: ___  
    Only extract what's clearly present. If something is missing, leave it blank.
    
    Lead with the statement "This is the personal information we were able to extract from your resume. 
    If anything is missing, there may be an issue with your formatting: \n\n" \
    \n\n{resume}
    """

)
info_chain = LLMChain(llm=llm, prompt=info_prompt, output_key="personal_info")

# Job Role Suggestion
roles_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    You are a professional, friendly, and detail-oriented career assistant. \
    Speak directly to the candidate using “you.” Be specific, constructive, and easy to understand. \
    Based on the resume below, suggest 5 to 10 job roles the person is suited for. \
    List the name of the job role, and a quick 1-2 sentence explanation of the role and fit. \
    Lead with the statement "Here are some potential job roles you could pursue based on your resume: \n" \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    Here is the resume:\n\n{resume}
    """

    
)
roles_chain = LLMChain(llm=llm, prompt=roles_prompt, output_key="job_roles")

# Strengths
strengths_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    You are a professional, friendly, and detail-oriented career assistant. \
    Speak directly to the candidate using “you.” Be specific, constructive, and easy to understand. \
    Based on the resume, list 3–5 specific strengths \
    of the candidate and their resume. Each item should be 1–2 sentences.  
    Focus on stand-out traits, accomplishments, or qualities that add value. Avoid vague or generic praise. \
    Lead with the statement "These are some of your personal stengths as an applicant: \n" \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    \n\n{resume}
    """
)
strengths_chain = LLMChain(llm=llm, prompt=strengths_prompt, output_key="strengths")

# Improvements
improve_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    You are a professional, friendly, and detail-oriented career assistant. \
    Speak directly to the candidate using “you.” Be specific, constructive, and easy to understand. \
    Suggest 2–5 specific improvements for the resume. \
    These can address clarity, completeness, or content. Avoid suggestions about formatting or dates in the future.  
    Be practical and constructive — no vague advice like 'be more specific.' 
    Do not suggest visual layout or formatting changes. Do not suggest improvements involving timing of dates. 
    If the resume is already strong, say so. Don't make up any problems that aren't really there. \
    Lead with the statement "These are some improvements you could make to enhance your resume: \n" \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    \n\n{resume}
    """

)
improve_chain = LLMChain(llm=llm, prompt=improve_prompt, output_key="improvements")

# Career Suggestions
tips_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    You are a professional, friendly, and detail-oriented career assistant. \
    Speak directly to the candidate using “you.” Be specific, constructive, and easy to understand. \
    Based on this resume, provide 3–5 personalized career tips. \
    These may include project ideas, ways to gain experience, resources to explore, or networking advice.  
    Tips should be actionable, relevant, and easy to understand. Only include suggestions that would actually benefit them \
    and that they have not yet already completed. Help them land a job! \
    Lead with the statement "Here's a list of career tips you could implement to improve your chances of being hired: \n" \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    \n\n{resume}
    """
)
tips_chain = LLMChain(llm=llm, prompt=tips_prompt, output_key="career_tips")

# Spelling Check
spelling_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
    Check this resume for any spelling/grammatical errors. If there are errors, state where the error \
    is and the necessary change. If there are no errors, simply output 'Good job! No spelling errors detected.' \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    Format your response nicely with markdown. \
    Here is the resume:\n\n{resume}
    """

)
spelling_chain = LLMChain(llm=llm, prompt=spelling_prompt, output_key="spelling")






############ Sequential chain #################



# Combine chains into a sequential pipeline
pipeline = SequentialChain(
    chains=[summary_chain, rating_chain, info_chain, roles_chain, strengths_chain, tips_chain, improve_chain, spelling_chain],
    input_variables=["resume"],
    output_variables=["summary", "rating", "personal_info", "job_roles", "strengths", "career_tips", 
                      "improvements", "spelling"],
    verbose=True
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from uploaded PDF bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    if doc.page_count > 3:
        raise ValueError("Your resume exceeds the 3-page limit. Please upload a shorter version.")
    
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def analyze_resume(file_bytes: bytes) -> dict:
    """Run the LangChain pipeline on extracted resume text."""
    resume_text = extract_text_from_pdf(file_bytes)
    return pipeline({"resume": resume_text})

