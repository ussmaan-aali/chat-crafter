import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_llm(temperature:int=0, model_name:str="llama-3.1-70b-versatile"):
    llm = ChatGroq(
        model=model_name,
        temperature=temperature

    )

    return llm

def chain(retriever, prompt:str=None):

    llm = get_llm()
    if prompt is None:
        prompt = ChatPromptTemplate.from_template(
                """You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know.

                Context: {context}
                
                Question: {question}
                
                Helpful Answer:"""
            )
        
    qa_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain