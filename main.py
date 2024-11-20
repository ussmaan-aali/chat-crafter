import os
from typing import List
from src.chatbot_handler import chain
import streamlit as st
from langchain_community.document_loaders import TextLoader


# def load_documents(directory: str) -> List:
#     """
#     Load documents from a specified directory.

#     Args:
#         directory (str): Path to directory containing documents

#     Returns:
#         List of loaded documents
#     """
#     documents = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.txt'):
#             loader = TextLoader(os.path.join(directory, filename))
#             documents.extend(loader.load())
#     return documents



class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def load_text_file(uploaded_file):
    """Load text from a single uploaded file."""
    text = uploaded_file.getvalue().decode('utf-8')
    return Document(
        page_content=text, 
        metadata={'source': uploaded_file.name}
    )


def home():
    st.write("Hello from home")


def chatbot_creator():
    st.header("Chatbot Craft")

    chatbot_name = st.text_input("Enter the name for your chatbot:")
    system_prompt_place_holder = """You are a helpful AI assistant. Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"""
    system_prompt = st.text_area("Enter the system prompt for the chatbot", placeholder=system_prompt_place_holder)
    loaded_docs = st.file_uploader("Upload text documents for chatbot", type=['txt'], accept_multiple_files=True)

    if loaded_docs and chatbot_name:
        documents = [load_text_file(doc) for doc in loaded_docs]
        # Create RAG chatbot
        chatbot = chain(documents, chatbot_name, system_prompt)
        query = "When did the health for AWS kubernetes posted?"
        response = chatbot.invoke(query)
        st.write(response)

def main():
    page_names_to_funcs = {
        "Home": home,
        "Chatbot Creator": chatbot_creator,

    }

    page_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
    page_names_to_funcs[page_name]()
if __name__ == "__main__":
    main()