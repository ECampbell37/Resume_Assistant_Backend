# chatbot.py

import os
import fitz
from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

# Volatile in-memory chat history
USER_CHATBOTS: Dict[str, ConversationChain] = {}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

# Prompt template
system_prompt = """
You are a professional, friendly, and detail-oriented resume assistant chatbot. You are here to help a job applicant improve their resume and increase their chances of landing a job.

The user has uploaded the following resume. You should refer to its contents whenever answering their questions:

------------------ RESUME START ------------------
{resume}
------------------- RESUME END -------------------

Speak directly to the user using “you.” Offer practical, constructive, and specific feedback. Avoid fluff. Do not suggest visual formatting changes (assume this is plain text only).

If the user asks something outside the scope of the resume (e.g., interview tips), you can still answer, but stay helpful and relevant.
"""

human_prompt = "{input}"

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template(human_prompt),
])

def get_or_create_chatbot(user_id: str, resume_text: str) -> ConversationChain:
    if user_id in USER_CHATBOTS:
        return USER_CHATBOTS[user_id]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationChain(
        llm=llm,
        prompt=chat_prompt.partial(resume=resume_text),
        memory=memory,
        verbose=False,
    )
    USER_CHATBOTS[user_id] = chain
    return chain
