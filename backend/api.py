"""
api.py
------
API FastAPI exposant la recherche sémantique RAG via MongoDB Atlas Vector Search.

Routes :
  GET  /               → health check
  POST /search         → recherche sémantique (body: {"query": "..."})
  POST /upload_course  → ingestion dynamique d'un PDF (multipart/form-data)

Lancement :
  uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional

fitz = None  # importé conditionnellement plus bas
try:
    import fitz as _fitz  # PyMuPDF
    fitz = _fitz
except ImportError:
    pass  # l'erreur sera levée à l'appel de /upload_course si fitz est absent

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# pyrefly: ignore [missing-import]
from pymongo import MongoClient

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

# ── Chargement de l'environnement ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

MONGODB_URI     = os.getenv("MONGODB_URI")
DB_NAME         = "rag_university"
COLLECTION_NAME = "course_chunks"
VECTOR_INDEX    = "vector_index"
EMBEDDING_FIELD = "embedding"
MODEL_NAME      = "all-MiniLM-L6-v2"
TOP_K           = 3

# ── Paramètres du chunker (ingestion PDF) ────────────────────────────────────
CHUNK_SIZE    = 500   # nombre de mots par chunk
CHUNK_OVERLAP = 50   # chevauchement (en mots) entre deux chunks consécutifs

# ── Vérification au démarrage ─────────────────────────────────────────────────
if not MONGODB_URI:
    sys.exit("[ERROR] MONGODB_URI introuvable dans .env -- serveur arrete.")

# ── Initialisation du modèle et de MongoDB (une seule fois au démarrage) ──────
print(f"[*] Chargement du modele '{MODEL_NAME}'...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("[OK] Modele pret.")

print("[*] Connexion a MongoDB Atlas...")
mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000)
collection   = mongo_client[DB_NAME][COLLECTION_NAME]
print(f"[OK] Connecte a '{DB_NAME}.{COLLECTION_NAME}'.")

# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="RAG University API",
    description="API de recherche sémantique sur les cours NoSQL via MongoDB Atlas Vector Search.",
    version="1.0.0",
)

# CORS — autorise le frontend local (React/Next.js sur localhost:3000 ou 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # En production : remplace par l'URL exacte du frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K        # nombre de résultats (défaut: 3)
    source_file: Optional[str] = None   # filtrer par fichier source (optionnel)


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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health Check")
def root():
    """Vérifie que l'API est opérationnelle."""
    return {
        "status": "ok",
        "message": "RAG University API est en ligne 🚀",
        "docs": "/docs",
    }


@app.post("/search", response_model=SearchResponse, summary="Recherche sémantique")
def search(request: SearchRequest):
    """
    Reçoit une question (query) en texte libre, la vectorise avec
    all-MiniLM-L6-v2, puis interroge MongoDB Atlas via $vectorSearch
    pour retourner les chunks les plus pertinents.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La query ne peut pas être vide.")

    limit = max(1, min(request.top_k, 10))  # borne entre 1 et 10

    # 1. Vectorisation de la question
    query_vector = embedding_model.encode(request.query).tolist()

    # 2. Pipeline $vectorSearch (+ filtre source optionnel)
    pipeline: list = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": limit * 20,  # plus de candidats pour compenser le filtre
                "limit": limit * 4 if request.source_file else limit,
            }
        },
    ]

    # Filtre post-vectorSearch sur le fichier source si fourni
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
            detail=f"Erreur MongoDB lors de la recherche : {str(e)}"
        )

    # 3. Formatage de la réponse
    results = [ChunkResult(**doc) for doc in raw_results]

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
    )


# ── Helpers internes ─────────────────────────────────────────────────────────
def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extrait le texte brut d'un PDF (bytes) page par page via PyMuPDF.
    Retourne le texte complet concaténé.
    """
    if fitz is None:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF (fitz) n'est pas installé. Exécute : pip install pymupdf",
        )
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages_text = [page.get_text("text") for page in doc]
    return "\n".join(pages_text)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Découpe un texte en chunks de `chunk_size` mots avec un chevauchement
    de `overlap` mots entre deux chunks consécutifs.
    """
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
        start = end - overlap  # recul pour le chevauchement
    return chunks


# ── Nouvelle route : ingestion PDF ───────────────────────────────────────────
@app.post("/upload_course", response_model=UploadResponse, summary="Ingestion d'un PDF")
async def upload_course(file: UploadFile = File(...)):
    """
    Reçoit un fichier PDF, extrait son texte, le découpe en chunks,
    vectorise chaque chunk avec all-MiniLM-L6-v2 (déjà chargé en mémoire)
    et insère les documents dans la collection MongoDB course_chunks.

    Structure de chaque document inséré :
    {
        chunk_id   : str   (UUID v4 unique)
        text       : str   (contenu du chunk)
        source_file: str   (nom du fichier PDF uploadé)
        embedding  : list  (vecteur 384 dimensions)
    }
    """
    # ── Validation du type de fichier ─────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers PDF sont acceptés.",
        )

    # ── Lecture du fichier en mémoire ─────────────────────────────────────
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de lire le fichier uploadé : {e}",
        )

    # ── Extraction du texte ───────────────────────────────────────────────
    try:
        full_text = _extract_text_from_pdf(pdf_bytes)
    except HTTPException:
        raise  # on remonte l'erreur déjà formatée
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Erreur lors de l'extraction du texte PDF : {e}",
        )

    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Le PDF ne contient pas de texte extractible (PDF scanné ?).",
        )

    # ── Chunking ──────────────────────────────────────────────────────────
    chunks = _chunk_text(full_text)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Aucun chunk généré depuis le PDF.",
        )

    # ── Vectorisation (batch pour la performance) ─────────────────────────
    try:
        vectors = embedding_model.encode(chunks, show_progress_bar=False).tolist()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la vectorisation : {e}",
        )

    # ── Construction des documents MongoDB ───────────────────────────────
    documents = [
        {
            "chunk_id":    str(uuid.uuid4()),
            "text":        chunk,
            "source_file": file.filename,
            EMBEDDING_FIELD: vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    # ── Insertion en base ─────────────────────────────────────────────────
    try:
        result = collection.insert_many(documents, ordered=False)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur MongoDB lors de l'insertion : {e}",
        )

    inserted_count = len(result.inserted_ids)
    return UploadResponse(
        filename=file.filename,
        chunks_inserted=inserted_count,
        message=f"{inserted_count} chunk(s) indexé(s) avec succès depuis '{file.filename}'.",
    )
