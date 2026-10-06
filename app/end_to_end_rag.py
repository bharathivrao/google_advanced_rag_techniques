import os
import logging
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_community.vectorstores import PGVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
logging.basicConfig(level=logging.ERROR)

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
# We use the recursive collection as it generally provides the best context boundaries
COLLECTION_NAME = f"{os.getenv('PGVECTOR_COLLECTION_NAME')}_recursive"

def getconn():
    instance_conn = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
    with Connector() as connector:
        return connector.connect(
            instance_conn, "pg8000",
            user=os.getenv("SQL_USER"), password=os.getenv("SQL_PASSWORD"),
            db=os.getenv("SQL_DATABASE_NAME"), ip_type=IPTypes.PUBLIC
        )

def main():
    print("--- Initializing Production RAG Pipeline ---")

    # 1. Setup Embeddings (Gemini Embedding 001)
    # We use this to vectorize the user's query to match our database.
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)

    # 2. Connect to Vector Store
    pg_conn_str = f"postgresql+pg8000://{os.getenv('SQL_USER')}:{os.getenv('SQL_PASSWORD')}@placeholder/{os.getenv('SQL_DATABASE_NAME')}"
    store = PGVector(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        connection_string=pg_conn_str,
        engine_args={"creator": getconn}
    )

    # 3. Setup The 'Filter Funnel' (Retriever + Reranker)
    # Step A: Fast retrieval of top 10 similar documents
    base_retriever = store.as_retriever(search_kwargs={"k": 10})

    # Step B: Precise reranking to find the top 3 most relevant
    reranker = VertexAIRank(
        project_id=PROJECT_ID,
        location_id="global", 
        ranking_config="default_ranking_config",
        title_field="source",
        top_n=3
    )

    # Combine A and B into a single retrieval object
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    # 4. Setup LLM (Gemini 3.5 Flash)
    # We use a low temperature (0.1) to reduce creativity and increase factual adherence.
    llm = VertexAI(model_name="gemini-3.5-flash", project=PROJECT_ID, location=REGION, temperature=0.1)

    # --- Execution Loop ---
    user_query = "Who is Harry Potter?"
    print(f"\nUser Query: {user_query}")
    print("Retrieving and Reranking documents...")

    # Retrieve the most relevant documents
    top_docs = compression_retriever.invoke(user_query)

    if not top_docs:
        print("No relevant documents found.")
        return

    # Build the Context String
    # We stitch the documents together, labeling them as Source 1, Source 2, etc.
    context_str = "\n\n".join([f"Source {i+1}: {d.page_content}" for i, d in enumerate(top_docs)])

    print(f"Found {len(top_docs)} relevant context chunks.")

    # 5. The Grounded Prompt
    template = """You are a helpful assistant. Answer the question strictly based on the provided context.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create the chain: Prompt -> LLM
    chain = prompt | llm

    print("Generating Answer via Gemini 3.5 Flash...")
    final_answer = chain.invoke({"context": context_str, "question": user_query})

    print(f"\nFINAL ANSWER:\n{final_answer}")

if __name__ == "__main__":
    main()