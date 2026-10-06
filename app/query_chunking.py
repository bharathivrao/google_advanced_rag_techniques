import os
import logging
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()
logging.basicConfig(level=logging.ERROR) # Only show errors to keep output clean

# Config
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
BASE_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME")

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

def main():
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)
    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"

    query = "Tell me about the Dursleys and their relationship with Harry Potter"
    print(f"\nQUERY: {query}\n" + "="*50)

    strategies = ["character", "recursive", "token"]

    for strategy in strategies:
        collection = f"{BASE_COLLECTION_NAME}_{strategy}"
        print(f"\nSTRATEGY: {strategy.upper()}")

        store = PGVector(
            collection_name=collection,
            embedding_function=embeddings,
            connection_string=pg_conn_str,
            engine_args={"creator": getconn}
        )

        results = store.similarity_search_with_score(query, k=2)
        for i, (doc, score) in enumerate(results):
            print(f"  Result {i+1} (Score: {score:.4f}): {doc.page_content[:150].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    main()