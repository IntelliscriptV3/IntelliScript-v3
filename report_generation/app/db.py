from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()  # 👈 make sure this line is present and BEFORE you read os.getenv

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("DB_URL not set. Check your .env file.")
# print(" Connected to:", os.getenv("DB_URL"))

DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
