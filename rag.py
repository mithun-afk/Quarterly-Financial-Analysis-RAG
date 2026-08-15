import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ingest import get_vectorstore

load_dotenv()

SYSTEM_PROMPT = """You are a precise financial research assistant analyzing quarterly financial announcements and press releases.

Rules:
1. Answer the user's question ONLY using the factual context provided below.
2. If the answer cannot be found in the provided context, reply exactly with:
   "The information is not available in the uploaded documents."
3. Do not assume, extrapolate, or invent figures, dates, or commentary.
4. When stating numbers (revenues, margins, profits), be exact as shown in the text.

Context:
{context}
"""

def generate_financial_answer(question: str, top_k: int = 5) -> dict:
    vectorstore = get_vectorstore()
    
    collection = vectorstore._collection
    if collection.count() == 0:
        return {
            "answer": "No documents have been indexed yet. Please upload and index quarterly financial PDFs first.",
            "sources": []
        }

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    formatted_context = ""
    sources = []
    seen_sources = set()

    for doc in retrieved_docs:
        file_name = doc.metadata.get("source_file", "Unknown File")
        page_num = doc.metadata.get("page_number", doc.metadata.get("page", 1))
        content = doc.page_content
        
        formatted_context += f"\n--- Source: {file_name} (Page {page_num}) ---\n{content}\n"
        
        source_key = (file_name, page_num)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({"file": file_name, "page": page_num})

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    # Swap OpenAI for Groq (using LLaMA 3 8B or Mixtral)
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.1
    )
    
    chain = prompt_template | llm

    response = chain.invoke({
        "context": formatted_context,
        "question": question
    })

    return {
        "answer": response.content.strip(),
        "sources": sources
    }