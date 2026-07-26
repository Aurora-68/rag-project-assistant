"""
api.py — fastapi backend for rag semantic search over mongodb atlas.

routes:
  GET  /               → health check
  POST /search         → semantic search (body: {"query": "..."})
  POST /upload_course  → pdf ingestion (multipart/form-data)

usage:
  uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional

fitz = None
try:
    import fitz as _fitz  # pymupdf
    fitz = _fitz
except ImportError:
    pass  # error raised at runtime if /upload_course is called without it

import certifi
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient  # pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer  # pyrefly: ignore [missing-import]

# load env vars from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

MONGODB_URI     = os.getenv("MONGODB_URI")
DB_NAME         = "rag_university"
COLLECTION_NAME = "course_chunks"
VECTOR_INDEX    = "vector_index"
EMBEDDING_FIELD = "embedding"
MODEL_NAME      = "all-MiniLM-L6-v2"
TOP_K           = 3

# chunking params for pdf ingestion
CHUNK_SIZE    = 500   # words per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

if not MONGODB_URI:
    sys.exit("[error] MONGODB_URI not found in .env — server stopped.")

# load embedding model once at startup
print(f"[*] loading model '{MODEL_NAME}'...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("[ok] model ready.")

# connect to mongodb atlas
print("[*] connecting to mongodb atlas...")
mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000, tlsCAFile=certifi.where())
collection   = mongo_client[DB_NAME][COLLECTION_NAME]
print(f"[ok] connected to '{DB_NAME}.{COLLECTION_NAME}'.")

# fastapi app
app = FastAPI(
    title="RAG University API",
    description="semantic search api for course materials via mongodb atlas vector search.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- pydantic schemas ---

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K
    source_file: Optional[str] = None


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    lecture_number: Optional[int] = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResult]
    total: int


class UploadResponse(BaseModel):
    filename: str
    chunks_inserted: int
    message: str


# --- routes ---

@app.get("/", summary="Health Check")
def root():
    """basic health check endpoint."""
    return {
        "status": "ok",
        "message": "rag university api is running",
        "docs": "/docs",
    }


@app.post("/search", response_model=SearchResponse, summary="Semantic Search")
def search(request: SearchRequest):
    """
    takes a text query, encodes it with all-MiniLM-L6-v2, and runs
    a $vectorSearch aggregation against mongodb atlas to return the
    most relevant chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty.")

    limit = max(1, min(request.top_k, 10))

    # vectorize the query
    query_vector = embedding_model.encode(request.query).tolist()

    # build $vectorSearch pipeline
    # when filtering by source_file we cast a wider net since post-filtering
    # reduces the result set
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": 1000 if request.source_file else limit * 20,
                "limit": 200 if request.source_file else limit,
            }
        },
    ]

    # optional post-filter on source file
    if request.source_file:
        pipeline.append({"$match": {"source_file": request.source_file}})

    pipeline += [
        {"$limit": limit},
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "text": 1,
                "source_file": 1,
                "page_number": 1,
                "lecture_number": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        raw_results = list(collection.aggregate(pipeline))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"mongodb search error: {str(e)}"
        )

    results = [ChunkResult(**doc) for doc in raw_results]

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
    )


# --- helpers ---

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """extract raw text from pdf bytes page by page using pymupdf."""
    if fitz is None:
        raise HTTPException(
            status_code=500,
            detail="pymupdf (fitz) is not installed. run: pip install pymupdf",
        )
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages_text = [page.get_text("text") for page in doc]
    return "\n".join(pages_text)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """split text into word-level chunks with overlap."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


# --- pdf ingestion route ---

@app.post("/upload_course", response_model=UploadResponse, summary="PDF Ingestion")
async def upload_course(file: UploadFile = File(...)):
    """
    accepts a pdf file, extracts text, chunks it, generates embeddings
    with all-MiniLM-L6-v2, and inserts the documents into mongodb.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="only pdf files are accepted.",
        )

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"failed to read uploaded file: {e}",
        )

    try:
        full_text = _extract_text_from_pdf(pdf_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"pdf text extraction error: {e}",
        )

    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail="pdf contains no extractable text (scanned pdf?).",
        )

    chunks = _chunk_text(full_text)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="no chunks generated from pdf.",
        )

    try:
        vectors = embedding_model.encode(chunks, show_progress_bar=False).tolist()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"embedding generation error: {e}",
        )

    documents = [
        {
            "chunk_id":    str(uuid.uuid4()),
            "text":        chunk,
            "source_file": file.filename,
            EMBEDDING_FIELD: vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    try:
        result = collection.insert_many(documents, ordered=False)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"mongodb insertion error: {e}",
        )

    inserted_count = len(result.inserted_ids)
    return UploadResponse(
        filename=file.filename,
        chunks_inserted=inserted_count,
        message=f"{inserted_count} chunk(s) indexed from '{file.filename}'.",
    )
