import os
import warnings

# LangChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")

# Define the GPT Model
llm_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.7, model=llm_model)

################## PROMPT & CHAIN ##################

rewrite_prompt = PromptTemplate(
    input_variables=["resume", "context"],
    template="""
You are an intelligent, helpful resume writing expert. Rewrite the following resume to improve clarity, impact, and professionalism.

Your task:
- Use strong action verbs and measurable outcomes where possible
- Condense verbose phrases without losing meaning
- Improve grammar and writing quality
- Maintain a clean, modern tone suitable for ATS systems
- Preserve all existing content (don’t hallucinate new info), unless the candidate's context below says otherwise
- Make changes that flow better, or that are more logically coherent or organized than the original
- Format the revised resume using **Markdown** with clear headings, bullet points, and sections

The candidate may provide optional context: instructions about the kind of revision they want, and extra facts about their background. Follow their instructions, even when they conflict with the guidelines above, and treat any facts they give as true and usable. If the context says "None provided", ignore it.

Instructions:
- Use markdown headers (e.g., `## Education`) for each major section
- Use bullet points for job responsibilities, projects, or skills
- Do NOT include frontmatter, metadata, or code blocks

Jazz up this resume and get creative!
(Do not talk to the user, only output the resume)

--- Original Resume ---
{resume}

--- Candidate Context ---
{context}

--- Rewritten Resume in Markdown ---
"""
)

rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

################## MAIN FUNCTION ##################

def rewrite_resume(resume_text: str, context: str = "") -> str:
    """Run the LangChain rewriting logic on resume text, with optional user context."""
    return rewrite_chain.run({
        "resume": resume_text,
        "context": context.strip() or "None provided",
    })