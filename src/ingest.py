import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    PDF_URL,
    PDF_PATH,
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def download_pdf():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(PDF_PATH):
        return
    response = requests.get(PDF_URL, timeout=60)
    response.raise_for_status()
    with open(PDF_PATH, "wb") as f:
        f.write(response.content)


def load_and_split():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_vector_store():
    download_pdf()
    chunks = load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    vector_store.persist()
    print(f"Ingested {len(chunks)} chunks into Chroma at {CHROMA_DIR}")


if __name__ == "__main__":
    build_vector_store()
