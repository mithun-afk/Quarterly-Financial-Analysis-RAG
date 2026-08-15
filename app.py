import streamlit as st
import os
from ingest import ingest_pdf_files, get_vectorstore
from rag import generate_financial_answer

st.set_page_config(
    page_title="Quarterly Financial Analysis RAG",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quarterly Financial Analysis RAG")
st.caption("Upload company quarterly reports, process chunks to disk, and query financial commentary with GPT-4o.")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Sidebar - Stats & Status
with st.sidebar:
    st.header("🗄️ Vector Database Status")
    vs = get_vectorstore()
    total_stored = vs._collection.count()
    st.metric("Total Indexed Chunks", total_stored)
    st.info("**Embedding:** text-embedding-3-small\n\n**LLM:** GPT-4o (temp: 0.1)\n\n**Chunk Size:** 1200 | **Overlap:** 150")

# Section 1: Upload & Index
st.subheader("1. Ingest Quarterly Result PDFs")
uploaded_files = st.file_uploader(
    "Choose 3–4 quarterly results PDFs (Press releases / Statements)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("⚡ Index Uploaded PDFs", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file before indexing.")
    else:
        saved_paths = []
        for file in uploaded_files:
            target_path = os.path.join(DATA_DIR, file.name)
            with open(target_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(target_path)
            
        with st.spinner("Chunking text, generating embeddings, and storing to ChromaDB..."):
            file_count, chunk_count = ingest_pdf_files(saved_paths)
            st.success(f"✅ Ingestion complete: {file_count} files processed, {chunk_count} chunks stored.")
            st.rerun()

st.divider()

# Section 2: Financial Q&A
st.subheader("2. Ask Financial Analysts' Questions")
user_query = st.text_input("Enter your question (e.g., 'What was the operating margin trend across quarters?'):")

if st.button("Search & Answer"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant quarterly chunks and formulating answer..."):
            result = generate_financial_answer(user_query, top_k=5)
            
            st.markdown("### 📝 Answer")
            st.write(result["answer"])
            
            if result.get("sources"):
                st.markdown("### 🔍 Verified Sources & Page References")
                for s in result["sources"]:
                    st.markdown(f"- 📄 **`{s['file']}`** — Page **{s['page']}**")