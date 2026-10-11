import os
import logging
import sqlalchemy

from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes


load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
INSTANCE_NAME = os.getenv("SQL_INSTANCE_NAME")

DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")

INSTANCE_CONNECTION_NAME = (
    f"{PROJECT_ID}:{REGION}:{INSTANCE_NAME}"
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
        password=DB_PASS,
        db=DB_NAME,
        ip_type=IPTypes.PUBLIC,
    )


# ---------------------------------------------------------
# Enable pgvector
# ---------------------------------------------------------

def enable_pgvector():

    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    try:

        with engine.begin() as conn:

            conn.execute(
                sqlalchemy.text(
                    "CREATE EXTENSION IF NOT EXISTS vector"
                )
            )

            logging.info(
                "pgvector extension is enabled."
            )

    finally:

        engine.dispose()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    try:

        logging.info(
            "Setting up database %s",
            DB_NAME,
        )

        enable_pgvector()

    finally:

        connector.close()