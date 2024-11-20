from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
import chromadb


def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Split documents into smaller chunks.

    Args:
        documents (List): List of documents to split
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks

    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks: List, collection_name: str = "rag_collection") -> Chroma:
    """
    Create a vector store from document chunks.

    Args:
        chunks (List): List of document chunks
        collection_name (str): Name for the Chroma collection

    Returns:
        Chroma vector store
    """
    # Load MiniLM model
    ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

    class DefChromaEF(Embeddings):
        def __init__(self,ef):
            self.ef = ef

        def embed_documents(self,texts):
            return self.ef(texts)

        def embed_query(self, query):
            return self.ef([query])[0]

    
    chroma_client = chromadb.PersistentClient(path="./chroma")
    
    collection = chroma_client.get_or_create_collection(collection_name)

    db = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=DefChromaEF(ef))
    retriever = db.as_retriever(search_kwargs={"filter":{"id":"1"}})
    return retriever


    # # embeddings = OpenAIEmbeddings()
    # return Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embedding_function,
    #     collection_name=collection_name,
    #     client_settings=client_settings,
    # )

def create_rag_chain(vector_store: Chroma, system_prompt: str=None):
    """
    Create a RAG retrieval chain using Groq LLM.

    Args:
        vector_store (Chroma): Initialized vector store

    Returns:
        RAG retrieval chain
    """
    retriever = vector_store

    llm = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768"
    )

    if system_prompt is None:
        system_prompt = """
        You are a helpful AI assistant. Answer the question based only on the following context:
        {context}

        Question: {question}
        """
    prompt_template = ChatPromptTemplate.from_template(system_prompt)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain

def chain(documents: List, chatbot_name: str, system_prompt: str):
    """
    Main function to create RAG chatbot pipeline.

    Args:
        documents_directory (str): Path to directory with documents
    """
    document_chunks = split_documents(documents)

    # Create vector store
    vector_store = create_vector_store(document_chunks, collection_name=chatbot_name)

    # Create RAG chain
    rag_chain = create_rag_chain(vector_store, system_prompt)

    return rag_chain

