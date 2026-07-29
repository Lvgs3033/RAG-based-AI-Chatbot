# Agentic AI eBook — RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions **strictly**
based on the Konverge.ai Agentic AI eBook

Built entirely with free, open-source components — no paid APIs or services.

## Tech Stack

| Layer            | Technology                                      |
|------------------|--------------------------------------------------|
| Orchestration    | LangGraph (StateGraph)                          |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2 (local)  |
| Vector Database  | ChromaDB (local, free, persistent)              |
| LLM              | google/flan-t5-base (local, via Hugging Face `transformers`) |
| PDF Parsing      | pypdf / LangChain PyPDFLoader                   |
| API              | FastAPI                                         |
| UI (optional)    | Streamlit                                       |

All models run locally on your machine (CPU is fine, GPU is faster). No API
keys, no Pinecone account, no OpenAI/Anthropic keys required.

## Architecture

```
                 ┌───────────────────────┐
                 │   Ebook PDF (source)   │
                 └───────────┬────────────┘
                             │ download + parse
                             ▼
                 ┌───────────────────────┐
                 │  Chunking (LangChain)  │
                 └───────────┬────────────┘
                             │ embed
                             ▼
                 ┌───────────────────────┐
                 │  ChromaDB Vector Store │
                 └───────────┬────────────┘
                             │
   User Question ──────────►│
                             ▼
                 ┌───────────────────────┐
                 │  LangGraph: retrieve   │  ← similarity search, top-k chunks
                 └───────────┬────────────┘
                             ▼
                 ┌───────────────────────┐
                 │  LangGraph: generate   │  ← flan-t5-base answers only from
                 └───────────┬────────────┘    retrieved context
                             ▼
        ┌────────────────────────────────────┐
        │ Answer + Retrieved Chunks + Score   │
        └────────────────────────────────────┘
```

The `generate` node computes a confidence score from the vector similarity
of the retrieved chunks. If the best match falls below a threshold, the bot
refuses to answer instead of guessing, keeping responses grounded in the
document.

## Project Structure

```
rag-chatbot/
├── data/                # Downloaded PDF is stored here
├── chroma_db/            # Persisted vector store (created after ingest)
├── src/
│   ├── config.py         # Paths and model settings
│   ├── ingest.py         # Download PDF, chunk, embed, store in Chroma
│   ├── graph.py           # LangGraph RAG pipeline
│   ├── api.py             # FastAPI chat endpoint
│   └── ui.py               # Streamlit chat UI
├── requirements.txt
├── sample_queries.md
└── README.md
```

## Setup Instructions

### 1. Clone and install dependencies
```bash
git clone https://github.com/Lvgs3033/RAG-based-AI-Chatbot/edit/main
cd rag-chatbot
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ingest the eBook
```bash
cd src
python ingest.py
```

### 3. Run the Chat API
```bash
cd src
python api.py
```

### 4. Run the Streamlit UI 
```bash
cd src
streamlit run ui.py
```
## Sample Queries

See `sample_queries.md` for 6 example questions to try, including one
out-of-scope question that demonstrates the grounding/refusal behavior.
