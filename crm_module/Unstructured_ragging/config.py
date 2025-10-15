# config.py
import os
# ---------------- PATHS ----------------
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler-25.07.0\Library\bin"

# ---------------- MODELS ----------------
OPENAI_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------- CHUNKING ----------------
GROUP_SIZE = 3
TOP_TAGS = 5
TOP_K = 5 

# Base directory of this config.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_FILE = os.path.join(BASE_DIR, "FAISS_DB", "company_rules.index")
EMB_FILE = os.path.join(BASE_DIR, "FAISS_DB", "company_embeddings.npy")
META_FILE = os.path.join(BASE_DIR, "FAISS_DB", "company_metadata.json")

# ---------------- SOURCE FOLDER ----------------
SOURCE_FOLDER = "faiss_sources"
api_key = os.getenv("OPENAI_API_KEY")