import os
import logging
from dotenv import load_dotenv

from google.cloud.sql.connector import Connector, IPTypes

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_postgres import PGVector
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO)

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

DB_USER = os.getenv("SQL_USER")
DB_PASSWORD = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_NAME = os.getenv("SQL_INSTANCE_NAME")

BASE_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME")

COLLECTION_NAME = f"{BASE_COLLECTION_NAME}_recursive"

INSTANCE_CONNECTION_NAME = (
    f"{PROJECT_ID}:{REGION}:{INSTANCE_NAME}"
)

RANKING_LOCATION = os.getenv(
    "RANKING_LOCATION_ID",
    "global",
)


# ---------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------

required_config = {
    "PROJECT_ID": PROJECT_ID,
    "REGION": REGION,
    "SQL_USER": DB_USER,
    "SQL_PASSWORD": DB_PASSWORD,
    "SQL_DATABASE_NAME": DB_NAME,
    "SQL_INSTANCE_NAME": INSTANCE_NAME,
    "PGVECTOR_COLLECTION_NAME": BASE_COLLECTION_NAME,
}

missing = [
    key
    for key, value in required_config.items()
    if not value
]

if missing:
    raise RuntimeError(
        f"Missing required environment variables: "
        f"{', '.join(missing)}"
    )


# ---------------------------------------------------------
# Cloud SQL Connector
# ---------------------------------------------------------

connector = Connector()


def getconn():
    return connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        ip_type=IPTypes.PUBLIC,
    )


# ---------------------------------------------------------
# Format retrieved documents
# ---------------------------------------------------------

def format_documents(documents):

    formatted = []

    for index, doc in enumerate(documents, start=1):

        source = doc.metadata.get(
            "source",
            "unknown",
        )

        formatted.append(
            f"""[Source {index}]
Source: {source}
Content:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    try:

        print("--- Initializing RAG Pipeline ---")

        # -------------------------------------------------
        # 1. Embeddings
        # -------------------------------------------------

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            project=PROJECT_ID,
            location=REGION,
            vertexai=True,
        )

        # -------------------------------------------------
        # 2. Vector Store
        # -------------------------------------------------

        pg_conn_str = (
            f"postgresql+pg8000://"
            f"{DB_USER}:{DB_PASSWORD}"
            f"@placeholder/{DB_NAME}"
        )

        store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=pg_conn_str,
            engine_args={
                "creator": getconn,
                "pool_pre_ping": True,
            },
            use_jsonb=True,
        )

        # -------------------------------------------------
        # 3. Candidate Retrieval
        # -------------------------------------------------

        base_retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
            },
        )

        # -------------------------------------------------
        # 4. Vertex AI Reranking
        # -------------------------------------------------

        reranker = VertexAIRank(
            project_id=PROJECT_ID,
            location_id=RANKING_LOCATION,
            ranking_config="default_ranking_config",
            title_field="source",
            top_n=3,
        )

        retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=reranker,
        )

        # -------------------------------------------------
        # 5. LLM
        # -------------------------------------------------

        llm = ChatGoogleGenerativeAI(
            model="YOUR_CURRENT_GEMINI_FLASH_MODEL",
            project=PROJECT_ID,
            location=REGION,
            vertexai=True,
            temperature=0.1,
        )

        # -------------------------------------------------
        # 6. Grounded Prompt
        # -------------------------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a retrieval-grounded assistant.

Answer the user's question using only the supplied context.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information,
   say: "I don't know based on the provided context."
3. Do not invent facts.
4. Cite supporting sources using [Source N].
5. Keep the answer concise and factual.
""",
                ),
                (
                    "human",
                    """
Context:

{context}


Question:

{question}
""",
                ),
            ]
        )

        # Prompt -> Gemini -> string
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        # -------------------------------------------------
        # 7. Query
        # -------------------------------------------------

        user_query = "Who is Harry Potter?"

        print(f"\nUser Query: {user_query}")
        print("Retrieving candidate documents...")

        top_docs = retriever.invoke(user_query)

        if not top_docs:
            print(
                "No relevant documents were returned."
            )
            return

        print(
            f"Retrieved and reranked "
            f"{len(top_docs)} chunks."
        )

        # -------------------------------------------------
        # 8. Context
        # -------------------------------------------------

        context = format_documents(top_docs)

        # Useful while developing
        print("\n--- Retrieved Context ---")

        for i, doc in enumerate(
            top_docs,
            start=1,
        ):

            score = doc.metadata.get(
                "relevance_score",
                "N/A",
            )

            source = doc.metadata.get(
                "source",
                "unknown",
            )

            print(
                f"Source {i}: "
                f"score={score}, "
                f"source={source}"
            )

        # -------------------------------------------------
        # 9. Generate grounded answer
        # -------------------------------------------------

        print("\nGenerating answer...")

        answer = chain.invoke(
            {
                "context": context,
                "question": user_query,
            }
        )

        print("\nFINAL ANSWER:")
        print(answer)

    except Exception:

        logging.exception(
            "RAG pipeline failed."
        )

        raise

    finally:

        connector.close()


if __name__ == "__main__":
    main()