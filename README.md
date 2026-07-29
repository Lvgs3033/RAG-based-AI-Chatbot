# Agentic AI eBook — RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions **strictly**
based on the Konverge.ai Agentic AI eBook
(https://konverge.ai/pdf/Ebook-Agentic-AI.pdf).

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
git clone <your-repo-url>
cd rag-chatbot
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ingest the eBook

This downloads the PDF, chunks it, generates embeddings, and stores them in
a local ChromaDB instance.

```bash
cd src
python ingest.py
```

You should see a message like:
```
Ingested <N> chunks into Chroma at .../chroma_db
```

### 3. Run the Chat API

```bash
cd src
python api.py
```

The API will be available at `http://localhost:8000`.

Example request:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Agentic AI?"}'
```

Example response:
```json
{
  "answer": "Agentic AI refers to ...",
  "confidence": 0.62,
  "sources": [
    {
      "content": "...chunk text...",
      "source": "Ebook-Agentic-AI.pdf",
      "page": 2,
      "score": 0.62
    }
  ]
}
```

### 4. (Optional) Run the Streamlit UI instead

```bash
cd src
streamlit run ui.py
```

This opens a browser chat interface showing the answer, the confidence
score, and an expandable panel with the retrieved context chunks.

## How Grounding Is Enforced

1. The LLM prompt explicitly instructs the model to use only the provided
   context and to say it cannot find the answer otherwise.
2. A relevance-score threshold (`GROUNDING_THRESHOLD` in `config.py`) blocks
   generation entirely when no retrieved chunk is similar enough to the
   question, returning a fixed "not found in the ebook" response instead of
   letting the LLM improvise.

## Sample Queries

See `sample_queries.md` for 6 example questions to try, including one
out-of-scope question that demonstrates the grounding/refusal behavior.

## Notes

- First run will download the embedding model and the flan-t5-base model
  from Hugging Face (a few hundred MB); after that they are cached locally.
- To re-ingest after changing chunk size or the source PDF, delete the
  `chroma_db/` folder and re-run `python src/ingest.py`.
- Swap `LLM_MODEL` in `config.py` for any other local Hugging Face
  text2text/causal model if you want higher quality answers.
