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
    Summarize the resume below in 3–5 concise sentences. Make it easy for the reader to read. \
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

    ### Resume Evaluation Rubric

    1. **Personal Information**
    - Full name is included 
    - At least two contact methods (email, phone, LinkedIn, etc.) are clearly provided
    - Location (city, state, or general region) is present 

    2. **Summary Section**
    - Clear professional identity or career goal
    - Highlights strengths, experience, or goals
    - Professional and concise tone

    3. **Work Experience**
    - Job titles, companies, and dates are stated
    - Responsibilities and/or achievements are described
    - Specific or measurable examples are used
    - Career relevance or growth is shown

    4. **Education**
    - School name and degree/certification listed
    - Graduation date or timeframe included
    - Honors, awards, or academic highlights mentioned
    - Educational background is relevant or explained
    - Optional: relevant courses or training listed

    5. **Skills and Projects**
    - Relevant hard skills (tools, technologies, industry know-how) listed
    - Soft skills or workplace competencies included
    - Skills are relevant and non-repetitive
    - Projects (if present) include title, context, tools used
    - Projects show initiative or relevance

    6. **Writing Quality**
    - Spelling and grammar are strong
    - Professional, concise word choice
    - Action oriented language
    - Clear structure and logical section flow
    
    
    Score breakdown:
    - A resume of 90-100% should be a fantastic resume for a very marketable candidate
    - A resume of 80-89% should be a well structured resume for a reasonably good candidate
    - A resume of 70-79% should be a decent resume for a decent candidate, with some issues here or there
    - A resume of below 70% should have something significantly wrong with it (not enough qualifications, poor content, etc)

    ---

    First, write a line with the score:  
    **Overall Rating: __%**

    Then, after a divider, on the next line, highlight each area of the rubric and how the resume measures on each category. \
    Don't mention the rubric sections verbatim, but rather use them as a guide. \
    Try to be as hyperspecific and as personalized to the resume and candidate as possible. \
    Format your output using markdown. By the end, the cannidate should have a clear understanding of the quality of their resume. \
    End with a short, hyper-personalized rationale for the overall rating.
        
    Do not suggest visual layout or formatting changes. Do not grade based on visual clarity. 
    
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
    Use this format (one per line, bold the titles):  
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
    Try to be as hyperspecific and as personalized to the resume and candidate as possible. \
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
    of the candidate and their resume. Each item should be 1–2 sentences, with a bolded section header.  
    Focus on stand-out traits, accomplishments, or qualities that add value. Avoid vague or generic praise. \
    Try to be as hyperspecific and as personalized to the resume and candidate as possible. \
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
    Try to be as hyperspecific and as personalized to the resume and candidate as possible. \
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
    These may include project ideas, ways to gain experience, resources to explore, or networking advice. \
    Try to be as hyperspecific and as personalized to the resume and candidate as possible. \
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
    Check this resume for spelling/grammatical errors. If there are errors, state where the error \
    is and the necessary change. If you are listing errors, make sure to format in a visually appealing manner. \
    If there are no errors, simply output 'Good job! No spelling errors detected.' \
    Ignore minor phrasing innacuracies. Do not list spacing or formatting issues. Only typos. 
    Only output the necessary text for this task (don't open with a statement or close with a statement). \

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

