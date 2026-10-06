import os
import logging
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAI
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.basicConfig(level=logging.ERROR)

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
COLLECTION_NAME = f"{os.getenv('PGVECTOR_COLLECTION_NAME')}_recursive"

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

def generate_hyde_doc(query, llm):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Write a concise, hypothetical answer to the question. Question: {question} Answer:"
    )
    chain = prompt | llm
    return chain.invoke({"question": query})

def generate_step_back(query, llm):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Write a more general, abstract question that concepts in this question. Original: {question} Step-back:"
    )
    chain = prompt | llm
    return chain.invoke({"question": query})

def main():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        vertexai=True,
        project=PROJECT_ID, 
        location=REGION)

    llm = VertexAI(model_name="gemini-3.5-flash", 
                    project=PROJECT_ID, 
                    location=REGION, 
                    temperature=0.5)

    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"
    store = PGVector(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        connection_string=pg_conn_str,
        engine_args={"creator": getconn}
    )
    retriever = store.as_retriever(search_kwargs={"k": 2})

    original_query = "Tell me about the Dursleys."
    print(f"ORIGINAL QUERY: {original_query}\n" + "-"*30)

    # 1. HyDE
    hyde_doc = generate_hyde_doc(original_query, llm)
    print(f"HyDE Generated Doc: {hyde_doc.strip()[:100]}...")
    hyde_results = retriever.invoke(hyde_doc)
    print(f"HyDE Retrieval: {hyde_results[0].page_content[:100]}...\n")

    # 2. Step-back
    step_back_q = generate_step_back(original_query, llm)
    print(f"Step-back Query: {step_back_q.strip()}")
    step_results = retriever.invoke(step_back_q)
    print(f"Step-back Retrieval: {step_results[0].page_content[:100]}...")

if __name__ == "__main__":
    main()