from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from graph import run_query

app = FastAPI(title="Agentic AI eBook RAG Chatbot")


class ChatRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    content: str
    source: str
    page: Optional[int]
    score: float


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[SourceChunk]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = run_query(request.question)
    return ChatResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["documents"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
