# tailor.py

import warnings

# LangChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")

# Define the GPT Model
llm_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.6, model=llm_model)

################## PROMPT & CHAIN ##################

tailor_prompt = PromptTemplate(
    input_variables=["resume", "job_description", "context"],
    template="""
You are an expert resume writer. Rewrite the candidate's resume so it is laser-focused on ONE specific job, giving them the best possible chance of getting it.

You are given:
- The candidate's RESUME
- The JOB DESCRIPTION
- The candidate's CONTEXT: optional instructions about how they want to be presented, and extra facts about their background. Follow their instructions and treat their facts as true. If it says "None provided", ignore it.

THE CORE RULE: Include only what makes this candidate look like the right hire for THIS job. Apply this test to everything: sections, roles, projects, skills, bullets, and individual details like numbers and years of experience. Ask "Would the hiring manager for this job care about this?" If not, cut it.

How to handle details:
- RELEVANT details (numbers, scale, tools, outcomes that support something the employer is asking for): keep them exactly as written, and lead with them.
- IRRELEVANT or loosely related details (for example, years of experience in a field the job doesn't ask for, or a metric from unrelated work): leave them out, even from a bullet you keep. Don't headline them.
- Never change a number or replace a relevant specific with a vague phrase.

Tools:

1. CUT: Remove or shrink entries, details, skills, and sections that don't help for this job.
2. REORDER: Put the strongest evidence for this job first, both the sections and the entries within them. Projects can go above work experience, education can go last, and so on.
3. REWORD: Change the emphasis and vocabulary of a bullet to match the job description (lead with the part the employer cares about, or use the job's term for the same skill). Rewording changes how something is said, never what the facts are.
4. FRAME: Start with a short summary (2-3 sentences) that positions the candidate as exactly what this employer is looking for.

The summary:
- Build it only from strengths that match what this job asks for.
- Do not open with years of experience or other numbers unless they are in the area the job is hiring for. If experience is only loosely related, describe the relevant strength itself instead of the total years.
- Use the job's language and be specific, with no generic praise.

Rules:
- Only use facts from the resume or the candidate's context. Never invent skills, tools, titles, dates, or numbers. Reframing real experience is encouraged; making things up is not.
- No fluff. Do not add endings that explain why something matters (like "ensuring...", "demonstrating...", "showcasing...", "resulting in enhanced..."), and avoid vague words like "robust", "seamless", "dynamic", or "innovative". A bullet should state what was done and its real result, nothing more.
- Don't add metrics that aren't in the resume or context.
- Write like a normal resume: never use "I", "me", "my", "we", or "our".
- Always keep the contact info. Keep titles, employers, dates, and degrees accurate for everything you include.

Output format: Markdown. Start with a `#` header with the candidate's name and their contact line, then a `## Summary` section, then the remaining sections with `##` headers and bullet points. Output only the resume, with no commentary or code blocks.

--- RESUME ---
{resume}

--- CONTEXT ---
{context}

--- JOB DESCRIPTION ---
{job_description}

--- TAILORED RESUME ---
"""
)

tailor_chain = LLMChain(llm=llm, prompt=tailor_prompt)

################## MAIN FUNCTION ##################

def tailor_resume(resume_text: str, job_description: str, context: str = "") -> str:
    """Tailor resume text to a specific job description, using optional user context."""
    return tailor_chain.run(
        resume=resume_text,
        job_description=job_description,
        context=context.strip() or "None provided",
    )