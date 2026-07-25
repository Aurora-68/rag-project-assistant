"""
api.py
------
API FastAPI — Recherche sémantique RAG via MongoDB Atlas Vector Search.
Version : 2.0.0 — Serverless (compatible Vercel / toute plateforme légère)

Changements v2.0 :
  - PyTorch et sentence-transformers supprimés (trop lourds pour Vercel < 250 Mo).
  - Embedding externalisé vers l'API Hugging Face Inference :
      Modèle : sentence-transformers/all-MiniLM-L6-v2
      Dimensions : 384 — IDENTIQUES aux données MongoDB existantes.
      → Aucune réindexation nécessaire.
  - Variable d'environnement ajoutée : EMBEDDING_API_KEY (token HF : hf_...)

Routes :
  GET  /               → health check
  POST /search         → recherche sémantique  { query, top_k?, source_file? }
  POST /upload_course  → ingestion PDF dynamique (multipart/form-data)

Lancement local :
  uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import uuid
import time
import requests
from pathlib import Path
from typing import List, Optional

# ── PyMuPDF — import conditionnel ─────────────────────────────────────────────
fitz = None
try:
    import fitz as _fitz
    fitz = _fitz
except ImportError:
    print("⚠️  PyMuPDF (fitz) non installé — /upload_course sera inopérant.")

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient

# ── Variables d'environnement ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

MONGODB_URI       = os.getenv("MONGODB_URI")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")   # Token Hugging Face (hf_...)

DB_NAME         = "rag_university"
COLLECTION_NAME = "course_chunks"
VECTOR_INDEX    = "vector_index"
EMBEDDING_FIELD = "embedding"
TOP_K           = 3

# Modèle HF : même que celui utilisé pour générer les embeddings MongoDB
# → vecteurs 384 dims, espace vectoriel identique, aucune réindexation
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL  = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_ID}"

# ── Paramètres du chunker (upload) ────────────────────────────────────────────
CHUNK_SIZE    = 500   # mots par chunk
CHUNK_OVERLAP = 50    # mots de chevauchement

# ── Vérifications au démarrage ────────────────────────────────────────────────
if not MONGODB_URI:
    sys.exit("❌ MONGODB_URI introuvable dans .env — serveur arrêté.")

if not EMBEDDING_API_KEY:
    print("⚠️  EMBEDDING_API_KEY non définie — /search et /upload_course retourneront 500.")

# ── Connexion MongoDB ─────────────────────────────────────────────────────────
print("🔌 Connexion à MongoDB Atlas...")
mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000)
collection   = mongo_client[DB_NAME][COLLECTION_NAME]
print(f"✅ Connecté à '{DB_NAME}.{COLLECTION_NAME}'.")

# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="RAG University API",
    description=(
        "API de recherche sémantique sur les cours universitaires via MongoDB Atlas "
        "Vector Search. Embedding externalisé vers Hugging Face Inference API. "
        "Version Serverless — compatible Vercel."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schémas Pydantic ──────────────────────────────────────────────────────────
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


# ── Helpers : Embedding via API Hugging Face Inference ────────────────────────

def _parse_hf_response(data) -> List[List[float]]:
    """
    Normalise la réponse de l'API HF Feature Extraction en une liste de vecteurs.
    L'API peut retourner plusieurs formats selon le batch et la version du modèle.
    """
    if not data or not isinstance(data, list):
        raise ValueError(f"Réponse HF inattendue : {type(data)}")

    first = data[0]

    if isinstance(first, float):
        # Vecteur unique plat : [0.1, -0.3, ...] → [[0.1, -0.3, ...]]
        return [data]

    if isinstance(first, list):
        if isinstance(first[0], float):
            # Batch de vecteurs : [[0.1, ...], [0.2, ...], ...]
            return data
        if isinstance(first[0], list):
            # Format token-level imbriqué : [[[float, ...]], [[float, ...]]]
            # On prend le premier "token" (embedding de la phrase après pooling)
            return [item[0] for item in data]

    raise ValueError(f"Format de vecteur HF non reconnu. Premier élément : {type(first)}")


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Vectorise une liste de textes via l'API Hugging Face Inference.

    Modèle : sentence-transformers/all-MiniLM-L6-v2 → 384 dimensions.
    Identique au modèle local supprimé → données MongoDB 100% compatibles.

    Traite les textes par lots de 32 pour respecter les limites de l'API.
    Gère automatiquement les cold-starts du modèle (option wait_for_model).
    """
    if not EMBEDDING_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="EMBEDDING_API_KEY non configurée. Ajoutez-la dans les variables d'environnement.",
        )

    headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}"}
    all_vectors: List[List[float]] = []
    BATCH_SIZE = 32

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        payload = {
            "inputs": batch,
            "options": {"wait_for_model": True},  # Évite les 503 de cold-start
        }

        for attempt in range(3):
            try:
                resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

                if resp.status_code == 503:
                    # Modèle encore en chargement malgré wait_for_model
                    time.sleep(10)
                    continue

                resp.raise_for_status()
                vectors = _parse_hf_response(resp.json())
                all_vectors.extend(vectors)
                break

            except requests.RequestException as exc:
                if attempt == 2:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Erreur API Hugging Face (tentative {attempt+1}/3) : {exc}",
                    )
                time.sleep(5)

    return all_vectors


# ── Helpers : traitement PDF ──────────────────────────────────────────────────

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extrait le texte de toutes les pages d'un PDF (via PyMuPDF)."""
    if fitz is None:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF (fitz) n'est pas installé. Lancez : pip install PyMuPDF",
        )
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text("text") for page in doc]
    return "\n".join(pages)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Découpe un texte en chunks de `chunk_size` mots avec `overlap` mots de chevauchement."""
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health Check")
def root():
    """Vérifie que l'API est opérationnelle."""
    return {
        "status": "ok",
        "message": "RAG University API v2.0 (Serverless) est en ligne 🚀",
        "docs": "/docs",
        "embedding_model": HF_MODEL_ID,
        "embedding_dims": 384,
        "embedding_provider": "Hugging Face Inference API",
    }


@app.post("/search", response_model=SearchResponse, summary="Recherche sémantique")
def search(request: SearchRequest):
    """
    Vectorise la question via l'API HF Inference, puis interroge MongoDB Atlas
    via $vectorSearch. Filtre optionnel par source_file pour isoler un PDF.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La query ne peut pas être vide.")

    limit = max(1, min(request.top_k or TOP_K, 10))

    # 1. Vectorisation de la question via l'API externe
    query_vector = _embed_texts([request.query])[0]

    # 2. Pipeline MongoDB Aggregation
    pipeline: list = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": limit * 20,
                "limit": limit * 4 if request.source_file else limit,
            }
        },
    ]

    # Filtre optionnel : limiter aux chunks d'un PDF spécifique
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
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur MongoDB lors de la recherche : {exc}",
        )

    results = [ChunkResult(**doc) for doc in raw_results]
    return SearchResponse(query=request.query, results=results, total=len(results))


@app.post("/upload_course", response_model=UploadResponse, summary="Ingestion d'un PDF")
async def upload_course(file: UploadFile = File(...)):
    """
    Reçoit un fichier PDF, extrait le texte, découpe en chunks,
    vectorise via l'API HF Inference et insère les documents dans MongoDB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    # 1. Lecture du fichier en mémoire
    try:
        pdf_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le fichier : {exc}")

    # 2. Extraction du texte (PyMuPDF)
    try:
        full_text = _extract_text_from_pdf(pdf_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Erreur extraction PDF : {exc}")

    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Ce PDF ne contient pas de texte extractible (PDF scanné / image uniquement).",
        )

    # 3. Découpage en chunks
    chunks = _chunk_text(full_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Aucun chunk généré depuis ce PDF.")

    # 4. Vectorisation batch via API HF (mêmes 384 dims que les données existantes)
    try:
        vectors = _embed_texts(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur vectorisation : {exc}")

    # 5. Construction des documents MongoDB
    documents = [
        {
            "chunk_id":      str(uuid.uuid4()),
            "text":          chunk,
            "source_file":   file.filename,
            EMBEDDING_FIELD: vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    # 6. Insertion en base (ordered=False = robuste aux erreurs partielles)
    try:
        result = collection.insert_many(documents, ordered=False)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur MongoDB lors de l'insertion : {exc}",
        )

    inserted_count = len(result.inserted_ids)
    return UploadResponse(
        filename=file.filename,
        chunks_inserted=inserted_count,
        message=f"{inserted_count} chunk(s) indexé(s) avec succès depuis '{file.filename}'.",
    )
