import json
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from modules.file_utils import load_and_convert_to_documents
from modules.langchain_utils import (
    chunk_data, initialize_embeddings, initialize_vector_store, initialize_qa_chain
)
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)
logging.info("Starting the Flask application...")

# Initialize Pinecone
logging.info("Initializing Pinecone...")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Load and process documents
logging.info("Loading and processing documents...")
file_paths = [
    "dataset/Aetna_Test_Data.txt",
    "dataset/Athena_Test_Data.txt",
    "dataset/Epic_Test_Data.txt"
]

documents = load_and_convert_to_documents(file_paths)
logging.info(f"Loaded {len(documents)} documents.")
documents = chunk_data(documents)
logging.info(f"Chunked documents into {len(documents)} chunks.")

# Initialize embeddings and vector store
logging.info("Initializing embeddings...")
embeddings = initialize_embeddings(api_key=os.getenv('OPENAI_API_KEY'))

logging.info(f"Index '{os.getenv('INDEX_NAME')}' not found. Creating a new index.")
index = initialize_vector_store(documents, embeddings, index_name=os.getenv('INDEX_NAME'))

# Initialize QA Chain
logging.info("Initializing QA Chain...")
qa_chain = initialize_qa_chain(api_key=os.getenv('OPENAI_API_KEY'))

# Define retrieval and QA functions
def retrieve_query(query, k=2):
    logging.info(f"Retrieving query: '{query}' with top {k} results.")
    results = index.similarity_search(query, k=k)
    logging.info(f"Retrieved {len(results)} results.")
    return results

def retrieve_answers(query):
    system_prompt = (
        '''You are an expert data analyst specializing in finding anomalies, insights, outliers and summary from complex datasets. 
        For structured responses:
        {
        "response": {
            "heading": "<Provide a concise summary or title>",
            "content": "<Provide detailed information or analysis>"
        }
        }

        For simple responses:
        {
        "response": "<Provide detailed information or analysis>"
        }

        Ensure all responses strictly adhere to this format. Do not include any extra text or deviate from the structure.
        '''
        
    )
    
    # Construct the full prompt to send to the model
    user_prompt = f"User Query: {query}\nPlease analyze the data and respond accordingly."

    # Combine system and user prompts
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    logging.info(f"Retrieving answers for query: '{full_prompt}'")
    doc_search = retrieve_query(query)
    logging.info(f"Found {len(doc_search)} documents for query.")
    response = qa_chain.run(input_documents=doc_search, question=full_prompt)
    logging.info(f"Generated response: {response}")
    return response

# Flask route
@app.route('/query', methods=['POST','GET'])
def query_insights():
    logging.info("Received request on /query endpoint.")
    if request.method == 'POST':
        # Get query from request body (JSON)
        data = request.json
        query = data.get('query', '')
    elif request.method == 'GET':
        # Get query from URL parameters
        query = request.args.get('query', '')
    if not query:
        logging.warning("Query is missing in the request.")
        return jsonify({"error": "Query is required"}), 400
    logging.info(f"Processing query: '{query}'")
    response = retrieve_answers(query)
    logging.info("Query processed successfully.")
    return response

if __name__ == '__main__':
    logging.info("Starting the Flask server...")
    app.run(debug=True, port=6660)
