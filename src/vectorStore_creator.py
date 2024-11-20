from typing import List

from langchain.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_data(files: List):
    document_chunks = []
    for file in files:
        loader = UnstructuredFileLoader(file)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
            )
        document_chunks.extend(text_splitter.split_documents(document))
    return document_chunks


def create_db(chunks, collection_name):
    embedding_function = HuggingFaceEmbeddings()
    Chroma.from_documents(
        collection_name=collection_name,
        documents=chunks,
        embedding=embedding_function,
        persist_directory="./chroma"
    )

def get_retriever(collection_name):
    embedding_function = HuggingFaceEmbeddings()
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory="./chroma",
        embedding_function=embedding_function
    )
    return vectordb.as_retriever()