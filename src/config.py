import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "Ebook-Agentic-AI.pdf")

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "agentic_ai_ebook"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
GROUNDING_THRESHOLD = 0.30
