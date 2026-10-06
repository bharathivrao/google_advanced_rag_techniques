import os
import logging
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import PGVector

# Reranking Imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank

load_dotenv()
logging.basicConfig(level=logging.ERROR)

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"

# IMPORTANT: Target the recursive collection created in ingest_data.py
COLLECTION_NAME = f"{os.getenv('PGVECTOR_COLLECTION_NAME')}_recursive"
RANKING_LOCATION = os.getenv("RANKING_LOCATION_ID")

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

    print(f"Connecting to collection: {COLLECTION_NAME}")
    store = PGVector(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        connection_string=pg_conn_str,
        engine_args={"creator": getconn}
    )

    query = "What are the Horcruxes?"
    print(f"QUERY: {query}\n")

    # 1. Base Retriever (Vector Search) - Fetch top 10
    base_retriever = store.as_retriever(search_kwargs={"k": 10})

    # 2. Reranker - Select top 3 from the 10
    reranker = VertexAIRank(
        project_id=PROJECT_ID,
        location_id=RANKING_LOCATION,
        ranking_config="default_ranking_config",
        title_field="source",
        top_n=3
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    # Execute
    try:
        reranked_docs = compression_retriever.invoke(query)

        if not reranked_docs:
            print("No documents returned. Check if the collection exists and is populated.")

        print(f"--- Top 3 Reranked Results ---")
        for i, doc in enumerate(reranked_docs):
            print(f"Result {i+1} (Score: {doc.metadata.get('relevance_score', 'N/A')}):")
            print(f"  {doc.page_content[:200]}...\n")
    except Exception as e:
        print(f"Error during reranking: {e}")

if __name__ == "__main__":
    main()