import os
from typing import List, Dict
import json
from src.chatbot_handler import chain
import streamlit as st
import src.vectorStore_creator as vsc
from src.chain import chain
import tempfile


def user_handler(data:Dict=None, filename:str="user_data.json"):
    # Ensure file exists with empty list
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump([], f)
    
    with open(filename, 'r') as f:
        user = json.load(f)
    
    if data:
        user.append(data)
        with open(filename, 'w') as f:
            json.dump(user, f, indent=4)
    
    return user


def load_docs(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        # Create a temporary file to pass to UnstructuredFileLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Load document using UnstructuredFileLoader
            all_chunks.extend(vsc.load_data([temp_file_path]))
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    return all_chunks


def home():
    st.header("Chatbot World")

    user = user_handler()
    
    if not user:
        st.write("Not bots. Create one from chatbot creator.")

    else:
        for bot in user:
            with st.expander(bot['chatbot_name']):
                retriver = vsc.get_retriever(bot['chatbot_name'])
                
                qa_chain = chain(retriver)

                query = st.text_input("I am here to help. Write youre question")

                if query:
                    response = qa_chain.invoke(query)
                    st.write(response)


def chatbot_creator():
    st.header("Chatbot Craft")

    chatbot_name = st.text_input("Enter the name for your chatbot:")
    uploaded_docs = st.file_uploader("Upload text documents for chatbot", type=['pdf'], accept_multiple_files=True)

    if uploaded_docs and chatbot_name:
        document_chunks = load_docs(uploaded_docs)
        vsc.create_db(document_chunks, chatbot_name)

        user_handler({"chatbot_name": chatbot_name})
        st.success("Chatbot created")

def main():
    page_names_to_funcs = {
    "Home": home,
    "Chatbot Creator": chatbot_creator,
}

    page_name = st.sidebar.radio("Choose a page", list(page_names_to_funcs.keys()))
    page_names_to_funcs[page_name]()

if __name__ == "__main__":
    main()