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

# ---------------- FAISS FILES ----------------
INDEX_FILE = os.path.join("FAISS_DB", "company_rules.index")
EMB_FILE = os.path.join("FAISS_DB", "company_embeddings.npy")
META_FILE = os.path.join("FAISS_DB", "company_metadata.json")

# ---------------- SOURCE FOLDER ----------------
SOURCE_FOLDER = "faiss_sources"
api_key = os.getenv("OPENAI_API_KEY")