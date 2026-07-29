import os
from typing import TypedDict, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langgraph.graph import StateGraph, END

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    GROUNDING_THRESHOLD,
)


class RAGState(TypedDict):
    question: str
    documents: List[dict]
    answer: str
    confidence: float


_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if not os.path.exists(CHROMA_DIR):
    raise RuntimeError(
        "Chroma DB not found. Run 'python src/ingest.py' first to build the vector store."
    )

_vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=CHROMA_DIR,
)

_generator = pipeline(
    "text2text-generation",
    model=LLM_MODEL,
    max_new_tokens=256,
)

SYSTEM_INSTRUCTIONS = (
    "You are a strict question answering assistant. Answer ONLY using the "
    "context below, which comes from an ebook about Agentic AI. "
    "If the answer is not present in the context, reply exactly: "
    "'I could not find this in the Agentic AI ebook.' Do not use outside knowledge.\n\n"
)


def retrieve_node(state: RAGState) -> RAGState:
    results = _vector_store.similarity_search_with_relevance_scores(
        state["question"], k=TOP_K
    )
    documents = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Ebook-Agentic-AI.pdf"),
            "page": doc.metadata.get("page", None),
            "score": round(float(score), 4),
        }
        for doc, score in results
    ]
    state["documents"] = documents
    return state


def generate_node(state: RAGState) -> RAGState:
    documents = state["documents"]

    if not documents:
        state["answer"] = "I could not find this in the Agentic AI ebook."
        state["confidence"] = 0.0
        return state

    top_score = max(d["score"] for d in documents)

    if top_score < GROUNDING_THRESHOLD:
        state["answer"] = "I could not find this in the Agentic AI ebook."
        state["confidence"] = round(top_score, 4)
        return state

    context_text = "\n\n".join(d["content"] for d in documents)
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}Context:\n{context_text}\n\n"
        f"Question: {state['question']}\nAnswer:"
    )

    output = _generator(prompt)[0]["generated_text"].strip()

    if not output:
        output = "I could not find this in the Agentic AI ebook."

    state["answer"] = output
    state["confidence"] = round(top_score, 4)
    return state


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_compiled_graph = build_graph()


def run_query(question: str) -> RAGState:
    initial_state: RAGState = {
        "question": question,
        "documents": [],
        "answer": "",
        "confidence": 0.0,
    }
    return _compiled_graph.invoke(initial_state)


if __name__ == "__main__":
    result = run_query("What is Agentic AI?")
    print(result["answer"])
    print(result["confidence"])
