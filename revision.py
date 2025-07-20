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
    input_variables=["resume"],
    template="""
You are an intelligent, helpful resume writing expert. Rewrite the following resume to improve clarity, impact, and professionalism.

Your task:
- Use strong action verbs and measurable outcomes where possible
- Condense verbose phrases without losing meaning
- Improve grammar and writing quality
- Maintain a clean, modern tone suitable for ATS systems
- Preserve all existing content (don’t hallucinate new info)
- Make changes that flow better, or that are more logically coherent or organized than the original
- Format the revised resume using **Markdown** with clear headings, bullet points, and sections

Instructions:
- Use markdown headers (e.g., `## Education`) for each major section
- Use bullet points for job responsibilities, projects, or skills
- Do NOT include frontmatter, metadata, or code blocks

Jazz up this resume and get creative!
(Do not talk to the user, only output the resume)

--- Original Resume ---
{resume}

--- Rewritten Resume in Markdown ---
"""
)

rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

################## MAIN FUNCTION ##################

def rewrite_resume(resume_text: str) -> str:
    """Run the LangChain rewriting logic on resume text."""
    return rewrite_chain.run({"resume": resume_text})
