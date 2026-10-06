import os
import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Config
project_id = os.getenv("PROJECT_ID")
region = os.getenv("REGION")
instance_name = os.getenv("SQL_INSTANCE_NAME")
db_user = os.getenv("SQL_USER")
db_pass = os.getenv("SQL_PASSWORD")
db_name = os.getenv("SQL_DATABASE_NAME")
instance_connection_name = f"{project_id}:{region}:{instance_name}"

print(f"Project ID: {project_id}")
print(f"Region: {region}")
print(f"Instance Name: {instance_name}")
print(f"DB User: {db_user}")
print(f"DB Name: {db_name}")
print(f"Instance Connection Name: {instance_connection_name}")
print(f"DB Password: {db_pass}")

# Initialize Connector object
connector = Connector()

def getconn():
    conn = connector.connect(
        instance_connection_name,
        "pg8000",
        user=db_user,
        password=db_pass,
        db=db_name,
        ip_type=IPTypes.PUBLIC,
    )
    return conn

def enable_pgvector():
    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )
    with pool.connect() as db_conn:
        # Check if extension exists
        result = db_conn.execute(sqlalchemy.text("SELECT extname FROM pg_extension WHERE extname = 'vector';")).fetchone()
        if result:
            logging.info("pgvector extension is already enabled.")
        else:
            logging.info("Enabling pgvector extension...")
            db_conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
            db_conn.commit()
            logging.info("pgvector extension enabled successfully.")

if __name__ == "__main__":
    try:
        enable_pgvector()
    finally:
        connector.close()