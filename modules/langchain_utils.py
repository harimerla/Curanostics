from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import os

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def initialize_embeddings(api_key):
    return OpenAIEmbeddings(api_key=api_key)

def initialize_vector_store(documents, embeddings, index_name):
    return LangchainPinecone.from_documents(documents, embeddings, index_name=index_name)

def initialize_qa_chain(api_key, model_name="gpt-3.5-turbo-instruct", temperature=0.5):
    llm = OpenAI(api_key=api_key, model_name=model_name, temperature=temperature)
    return load_qa_chain(llm, chain_type="stuff")
