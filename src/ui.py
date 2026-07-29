import streamlit as st
from graph import run_query

st.set_page_config(page_title="Agentic AI eBook Chatbot", layout="wide")
st.title("Agentic AI eBook — RAG Chatbot")
st.caption("Answers are strictly grounded in the Agentic AI ebook (konverge.ai)")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question about the Agentic AI ebook")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        result = run_query(question)
    st.session_state.history.append((question, result))

for question, result in reversed(st.session_state.history):
    st.markdown(f"### Q: {question}")
    st.markdown(f"**Answer:** {result['answer']}")
    st.markdown(f"**Confidence score:** {result['confidence']}")
    with st.expander("Retrieved context chunks"):
        for i, doc in enumerate(result["documents"], start=1):
            st.markdown(f"**Chunk {i}** (score: {doc['score']}, page: {doc['page']})")
            st.write(doc["content"])
    st.divider()
