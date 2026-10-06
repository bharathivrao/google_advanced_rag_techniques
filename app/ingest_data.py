import os
import json
import logging
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
BASE_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME")
BOOKS_JSON_URL = "https://storage.googleapis.com/github-repo/generative-ai/gemini/reasoning-engine/sample_data/harry_potter_books.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_DOCS_TO_PROCESS = 10 

# Database Connector
def getconn():
    with Connector() as connector:
        return connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
            ip_type=IPTypes.PUBLIC,
        )

def download_data():
    logging.info(f"Downloading data from {BOOKS_JSON_URL}...")
    response = requests.get(BOOKS_JSON_URL)
    return response.json()

def prepare_chunks(json_data, strategy):
    documents = []

    # Iterate through the downloaded data
    for entry in json_data[:MAX_DOCS_TO_PROCESS]:

        # --- JSON PARSING LOGIC ---
        # The data structure nests content inside 'kwargs' -> 'page_content'
        if "kwargs" in entry and "page_content" in entry["kwargs"]:
            content = entry["kwargs"]["page_content"]

            # Extract metadata if available, ensuring it's a dict
            metadata = entry["kwargs"].get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"source": "unknown"}

            # Add the strategy to metadata for tracking
            metadata["strategy"] = strategy
        else:
            continue

        if not content:
            continue

        # Choose the splitter based on the strategy
        if strategy == "character":
            splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n")
        elif strategy == "token":
            splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        else: # default to recursive
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        # Split the content into chunks
        chunks = splitter.split_text(content)

        # Create Document objects for each chunk
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))

    return documents

def main():
    logging.info("Initializing Embeddings...")
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)

    data = download_data()
    strategies = ["character", "recursive", "token"]

    # Connection string for PGVector (uses the getconn helper)
    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"

    for strategy in strategies:
        collection_name = f"{BASE_COLLECTION_NAME}_{strategy}"
        logging.info(f"--- Processing strategy: {strategy.upper()} ---")
        logging.info(f"Target Collection: {collection_name}")

        # Prepare documents with the specific strategy
        docs = prepare_chunks(data, strategy)

        if not docs:
            logging.warning(f"No documents generated for strategy {strategy}. Check data source.")
            continue

        logging.info(f"Generated {len(docs)} chunks. Uploading to Cloud SQL...")

        # Initialize the Vector Store
        store = PGVector(
            collection_name=collection_name,
            embedding_function=embeddings,
            connection_string=pg_conn_str,
            engine_args={"creator": getconn},
            pre_delete_collection=True # Clears old data for this collection before adding new
        )

        # Batch add documents
        store.add_documents(docs)
        logging.info(f"Successfully finished {strategy}.\n")

if __name__ == "__main__":
    main()