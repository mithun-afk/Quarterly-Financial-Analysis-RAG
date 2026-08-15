import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ingest import ingest_pdf_files, get_vectorstore, PERSIST_DIRECTORY, COLLECTION_NAME
from rag import generate_financial_answer

app = FastAPI(
    title="Finance RAG API Service",
    description="Backend service for indexing and querying quarterly financial reports",
    version="1.0.0"
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceItem(BaseModel):
    file: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    embedding_model: str
    llm_model: str

@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    saved_paths = []
    for upload in files:
        dest_path = os.path.join(UPLOAD_DIR, upload.filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        saved_paths.append(dest_path)

    num_files, num_chunks = ingest_pdf_files(saved_paths)
    return {"files": num_files, "chunks": num_chunks}

@app.post("/ask", response_model=QueryResponse)
async def ask_financial_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = generate_financial_answer(request.question, top_k=request.top_k)
    return result

@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    vectorstore = get_vectorstore()
    chunk_count = vectorstore._collection.count()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": chunk_count,
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o"
    }