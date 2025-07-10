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
    template="You are a professional career assistant, specializing in analyzing resumes. \
    Refer to the cannidate as 'you', not their actual name, as you are talking to them. \
    Summarize the following user-uploaded resume in 3-5 sentences:\n\n{resume}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# Rating
rating_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
You are a professional career assistant specializing in resume evaluation across all industries.

Rate the following resume on a scale of 0–100% based on the rubric below. You are speaking *to* the candidate, so refer to them as "you" (not by name).

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
**Rating: __%**

Then, on the next line, write a paragraph explaining your reasoning. Highlight strengths, weaknesses, and suggest specific areas to improve.

Here is the resume to evaluate:

{resume}
"""
)
rating_chain = LLMChain(llm=llm, prompt=rating_prompt, output_key="rating")

# Personal Info Extraction
info_prompt = PromptTemplate(
    input_variables=["resume"],
    template="Extract the name, email, phone number, location, and relavent links \
    (LinkedIn, Github, Personal Website, Portfolio, etc.) from the resume below. \
    Use this format: Name: ___ \n Email: ___ \n Phone Number: ___ \n Location: ___ \n LinkedIn: ____ \
    Here is the resume:\n\n{resume}"
)
info_chain = LLMChain(llm=llm, prompt=info_prompt, output_key="personal_info")

# Job Role Suggestion
roles_prompt = PromptTemplate(
    input_variables=["resume"],
    template="Based on the resume below, suggest 5 to 10 job roles the person is suited for. \
    List the name of the job role, and a quick 1-2 sentence explanation of the role and fit. \
    Refer to the cannidate as 'you', not their actual name, as you are talking to them. \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    Here is the resume:\n\n{resume}"
)
roles_chain = LLMChain(llm=llm, prompt=roles_prompt, output_key="job_roles")

# Strengths
strengths_prompt = PromptTemplate(
    input_variables=["resume"],
    template="You are a professional career assistant, specializing in analyzing resumes. \
    List 3–5 specific strengths of this resume, and of the cannidate themselves. Your answers can be a few sentences each. \
    Refer to the cannidate as 'you', not their actual name, as you are talking to them. \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    Here is the resume:\n\n{resume}"
)
strengths_chain = LLMChain(llm=llm, prompt=strengths_prompt, output_key="strengths")

# Improvements
improve_prompt = PromptTemplate(
    input_variables=["resume"],
    template="You are a professional career assistant, specializing in analyzing resumes. \
    Suggest a few (2-5) specific improvements for this resume (content, clarity, etc.). Your answers can be a few sentences each. \
    Do not include anything involving future or incorrect dates, as this is taking place after your training cutoff. \
    Be as helpful as possible, and stick to clear improvements, rather than vague ones. If the resume is good as is, say it. \
    Refer to the cannidate as 'you', not their actual name, as you are talking to them. \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    Here is the resume:\n\n{resume}"
)
improve_chain = LLMChain(llm=llm, prompt=improve_prompt, output_key="improvements")

# Career Suggestions
tips_prompt = PromptTemplate(
    input_variables=["resume"],
    template="You are a professional career assistant, specializing in analyzing resumes. \
    Offer a few career tips for the cannidate based on this resume (content, clarity, etc.). \
    Your answers can be a few sentences each. The career suggestions should help the person become a better cannidate, \
    and can include things like project ideas, ways to get more experience, resources to consider, etc. \
    Make your answer clear and easy to understand. \
    Refer to the cannidate as 'you', not their actual name, as you are talking to them. \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    Here is the resume:\n\n{resume}"
)
tips_chain = LLMChain(llm=llm, prompt=tips_prompt, output_key="career_tips")

# Spelling Check
spelling_prompt = PromptTemplate(
    input_variables=["resume"],
    template="Check this resume for any spelling/grammatical errors. If there are errors, state where the error \
    is and the necessary change. If there are no errors, simply output 'Good job! No spelling errors detected.' \
    Only output the necessary text for this task (don't open with a statement or close with a statement). \
    Here is the resume:\n\n{resume}"
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
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def analyze_resume(file_bytes: bytes) -> dict:
    """Run the LangChain pipeline on extracted resume text."""
    resume_text = extract_text_from_pdf(file_bytes)
    return pipeline({"resume": resume_text})

