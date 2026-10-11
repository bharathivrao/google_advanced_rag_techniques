import os
import logging
import requests
from dotenv import load_dotenv

from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from langchain_core.documents import Document


load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")

INSTANCE_CONNECTION_NAME = (
    f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
)

BASE_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME")

BOOKS_JSON_URL = (
    "https://storage.googleapis.com/github-repo/"
    "generative-ai/gemini/reasoning-engine/sample_data/"
    "harry_potter_books.json"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_DOCS_TO_PROCESS = 10


# ---------------------------------------------------------
# Cloud SQL Connector
# ---------------------------------------------------------

# Keep the Connector alive while SQLAlchemy uses its
# connections.
connector = Connector()


def getconn():
    return connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
        ip_type=IPTypes.PUBLIC,
    )


# ---------------------------------------------------------
# Download data
# ---------------------------------------------------------

def download_data():
    logging.info("Downloading data...")

    response = requests.get(
        BOOKS_JSON_URL,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# Chunk documents
# ---------------------------------------------------------

def prepare_chunks(json_data, strategy):
    documents = []

    for entry in json_data[:MAX_DOCS_TO_PROCESS]:

        if (
            "kwargs" not in entry
            or "page_content" not in entry["kwargs"]
        ):
            continue

        content = entry["kwargs"]["page_content"]

        if not content:
            continue

        original_metadata = entry["kwargs"].get(
            "metadata",
            {},
        )

        if not isinstance(original_metadata, dict):
            original_metadata = {
                "source": "unknown"
            }

        # Make a copy so we don't mutate the source metadata.
        metadata = original_metadata.copy()
        metadata["strategy"] = strategy

        if strategy == "character":

            splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separator="\n",
            )

        elif strategy == "token":

            splitter = TokenTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

        chunks = splitter.split_text(content)

        for chunk_number, chunk in enumerate(chunks):

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk"] = chunk_number

            documents.append(
                Document(
                    page_content=chunk,
                    metadata=chunk_metadata,
                )
            )

    return documents


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    try:

        logging.info("Initializing Vertex AI embeddings...")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            project=PROJECT_ID,
            location=REGION,
            vertexai=True,
        )

        data = download_data()

        strategies = [
            "character",
            "recursive",
            "token",
        ]

        # Host is a placeholder because SQLAlchemy uses
        # getconn() to obtain the actual Cloud SQL connection.
        pg_conn_str = (
            f"postgresql+pg8000://"
            f"{DB_USER}:{DB_PASS}"
            f"@placeholder/{DB_NAME}"
        )

        for strategy in strategies:

            collection_name = (
                f"{BASE_COLLECTION_NAME}_{strategy}"
            )

            logging.info(
                "Processing strategy: %s",
                strategy.upper(),
            )

            logging.info(
                "Target collection: %s",
                collection_name,
            )

            docs = prepare_chunks(
                data,
                strategy,
            )

            if not docs:

                logging.warning(
                    "No documents generated for %s",
                    strategy,
                )

                continue

            logging.info(
                "Generated %d chunks.",
                len(docs),
            )

            # IMPORTANT:
            # This is langchain_postgres.PGVector,
            # NOT langchain_community.vectorstores.PGVector.
            store = PGVector(
                embeddings=embeddings,
                collection_name=collection_name,
                connection=pg_conn_str,
                engine_args={
                    "creator": getconn
                },
                pre_delete_collection=True,
                use_jsonb=True,
            )

            store.add_documents(docs)

            logging.info(
                "Successfully indexed %s.",
                strategy,
            )

    finally:

        # Close the Cloud SQL Connector only after all
        # database operations have completed.
        connector.close()


if __name__ == "__main__":
    main()