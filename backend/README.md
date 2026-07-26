# backend

fastapi server that handles semantic search and pdf ingestion for the rag pipeline.

## files

| file | description |
|------|-------------|
| `api.py` | main fastapi app with `/search` and `/upload_course` routes |
| `extract_data.py` | extracts text from a pdf and splits it into overlapping chunks |
| `generate_embeddings.py` | generates 384d vectors for each chunk using all-MiniLM-L6-v2 |
| `upload_to_mongodb.py` | uploads vectorized chunks to mongodb atlas |
| `test_search.py` | standalone script to test vector search against the database |

## api routes

- `GET /` — health check
- `POST /search` — takes a `query` string, returns the most relevant chunks from mongodb
- `POST /upload_course` — accepts a pdf file, extracts text, chunks it, embeds it, and stores it in mongodb

## running

```bash
# from the project root (not from backend/)
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

## data pipeline

for offline indexing (without the web ui):

```bash
python -m backend.extract_data
python -m backend.generate_embeddings
python -m backend.upload_to_mongodb
```
