import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "finance_reports"

def get_embeddings():
    # Use a free, local HuggingFace model instead of OpenAI
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# ... (The rest of the ingest_pdf_files function remains exactly the same) ...

def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

def ingest_pdf_files(file_paths: list[str]) -> tuple[int, int]:
    """
    Loads PDF files, chunks text into 1200 characters with 150 overlap,
    generates embeddings, and persists them into ChromaDB.
    """
    all_documents = []
    
    for path in file_paths:
        if not os.path.exists(path):
            continue
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(path)
            # PyPDFLoader page is 0-indexed; convert to 1-indexed for citation
            doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
        all_documents.extend(docs)

    if not all_documents:
        return 0, 0

    # 1200 char size helps retain complete financial line-item tables
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_documents)

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )

    return len(file_paths), len(chunks)