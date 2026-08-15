# 📈 Financial RAG System for Quarterly Reports

**Developer:** Mithun S.
**Project:** HCLTech × ET Masterclass – AI Skills for the Future (Assignment 1)

## 📌 Project Overview
This project is a Retrieval-Augmented Generation (RAG) system built for an investment advisory research desk. It allows financial analysts to upload consecutive quarterly financial results (PDFs), ask plain-English questions, and receive precise answers backed by exact page and document citations. 

The system relies strictly on the provided context, implementing an "Honest Refusal" mechanism to prevent AI hallucinations when asked for data not present in the documents.

## 📊 Data Sources
*   **Company Chosen:** HCL Technologies Ltd.
*   **Documents Indexed:** 
    *   `Financial-Results-for-the-quarter-ended-June-30-2026.pdf`
    *   `audited-financial-results-for-the-quarter-ended-June-30-2025.pdf`
    *   *(Note: Additional quarterly reports can be placed in the `/data` folder)*

## 🛠️ Technical Architecture
*   **Language:** Python 3.10+
*   **User Interface:** Streamlit
*   **PDF Processing:** `PyPDFLoader`
*   **Orchestration:** LangChain
*   **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Replaced OpenAI due to API constraints)
*   **Vector Database:** ChromaDB (Persisted locally to `./chroma_db`)
*   **LLM:** Groq `llama-3.1-8b-instant` (Replaced OpenAI due to API constraints, temperature = 0.1)

## 🪓 Chunking Strategy
*   **Chunk Size:** 1200 characters
*   **Chunk Overlap:** 150 characters
*   **Reasoning:** Financial press releases contain wide tabular structures, balance sheets, and dense management commentary. A 1200-character window ensures that entire tabular rows and contextual margin statements remain intact within a single chunk, rather than being fragmented, which drastically improves retrieval accuracy.

## 🚀 Setup and Run Instructions

**1. Clone the repository:**
```bash
git clone [https://github.com/mithun-afk/Quarterly-Financial-Analysis-RAG.git](https://github.com/mithun-afk/Quarterly-Financial-Analysis-RAG.git)
cd finance-rag# Quarterly-Financial-Analysis-RAG
