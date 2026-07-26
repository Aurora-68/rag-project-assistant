# rag-project-assistant

a retrieval-augmented generation (rag) chatbot for university course materials. students can upload course pdfs and ask questions — answers are pulled directly from the actual content using vector search, not hallucinated.

built as a semester project for the nosql & big data course at ibn tofail university, faculte des sciences — kenitra.

## how it works

```
user question → embedding (all-MiniLM-L6-v2) → mongodb atlas vector search
    → top-k relevant chunks → llm prompt (openrouter) → streamed answer
```

1. course pdfs are chunked and embedded into 384-dimensional vectors
2. vectors are stored in mongodb atlas with a vector search index
3. when a student asks a question, it gets embedded and matched against stored chunks
4. the most relevant chunks are injected into the llm prompt as context
5. the llm generates an answer strictly from those chunks (no hallucination)

## tech stack

| layer | tech |
|-------|------|
| frontend | next.js, react |
| backend api | fastapi, python |
| database | mongodb atlas (vector search) |
| embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| llm | openrouter api |
| pdf parsing | pymupdf, pypdf2 |

## project structure

```
rag-project-assistant/
├── backend/              # fastapi server + data pipeline scripts
│   ├── api.py            # main api (search + pdf upload)
│   ├── extract_data.py   # pdf → chunks
│   ├── generate_embeddings.py  # chunks → vectors
│   ├── upload_to_mongodb.py    # vectors → mongodb
│   └── test_search.py   # standalone search test
├── frontend/             # next.js chat interface
│   ├── src/app/          # pages + api route
│   ├── src/components/   # react components
│   └── public/           # static assets
├── data/                 # sample pdf + generated json files
├── requirements.txt      # python dependencies
└── .env                  # mongodb connection string
```

## setup

### backend

```bash
# create virtual environment
python -m venv venv
source venv/bin/activate        # macos/linux
# venv\Scripts\activate         # windows

# install pytorch (cpu only) then dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# configure mongodb connection
# edit .env and add your MONGODB_URI

# start the api server
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

### frontend

```bash
cd frontend

# install node dependencies
npm install

# configure environment
# edit .env.local and add your OPENROUTER_API_KEY

# start the dev server
npm run dev
```

open http://localhost:3000 in your browser.

### initial data pipeline (optional)

if you want to pre-index a pdf from the command line instead of using the upload button:

```bash
python -m backend.extract_data
python -m backend.generate_embeddings
python -m backend.upload_to_mongodb
```

## environment variables

| file | variable | description |
|------|----------|-------------|
| `.env` | `MONGODB_URI` | mongodb atlas connection string |
| `frontend/.env.local` | `OPENROUTER_API_KEY` | openrouter api key |
| `frontend/.env.local` | `BACKEND_URL` | fastapi backend url (default: http://localhost:8000) |
| `frontend/.env.local` | `NEXT_PUBLIC_BACKEND_URL` | same, but accessible from browser |

## authors

- **mohammed nassiri** — frontend & llm integration
- **yassine esserdaoui** — backend & data pipeline
