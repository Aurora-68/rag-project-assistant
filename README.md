# 🎓 RAG Project Assistant

A Retrieval-Augmented Generation (RAG) chatbot designed for university students. Upload your course PDFs, ask questions, and get accurate answers pulled directly from the actual content — no hallucinations.

Built as a semester project for the NoSQL & Big Data course at **Ibn Tofail University**, Faculté des Sciences — Kénitra.

## ⚙️ How It Works

```
User Question → Embedding → MongoDB Atlas Vector Search → Top-K Chunks → LLM Prompt → Streamed Answer
```

1. Course PDFs are chunked into overlapping segments and embedded into 384-dimensional vectors.
2. Vectors are stored in MongoDB Atlas with a dedicated vector search index.
3. When a student asks a question, it gets embedded and matched against stored chunks using cosine similarity.
4. The most relevant chunks are injected into the LLM prompt as grounding context.
5. The LLM generates an answer strictly from those chunks, citing its sources.

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js, React |
| Backend API | FastAPI, Python |
| Database | MongoDB Atlas (Vector Search) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| LLM | OpenRouter API |
| PDF Parsing | PyMuPDF, PyPDF2 |

## 📁 Project Structure

```
rag-project-assistant/
├── backend/                        # FastAPI server + data pipeline scripts
│   ├── api.py                      # Main API — /search and /upload_course
│   ├── extract_data.py             # PDF → text chunks
│   ├── generate_embeddings.py      # Chunks → 384d vectors
│   ├── upload_to_mongodb.py        # Vectors → MongoDB Atlas
│   └── test_search.py              # Standalone vector search test
├── frontend/                       # Next.js chat interface
│   ├── src/app/api/chat/route.js   # Server-side RAG orchestration
│   ├── src/components/             # React UI components
│   └── public/                     # Static assets (logos, avatars)
├── data/                           # Sample PDF + pipeline output files
├── requirements.txt                # Python dependencies
└── .env                            # MongoDB connection string
```

## 🚀 Getting Started

### Backend

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate             # Windows

# Install PyTorch (CPU) then project dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Add your MongoDB URI to the .env file, then start the server
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install

# Add your OpenRouter API key to .env.local, then start the dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 📂 Offline Data Pipeline (Optional)

If you'd like to pre-index a PDF from the command line instead of using the upload button:

```bash
python -m backend.extract_data
python -m backend.generate_embeddings
python -m backend.upload_to_mongodb
```

## 🔑 Environment Variables

| File | Variable | Description |
|------|----------|-------------|
| `.env` | `MONGODB_URI` | MongoDB Atlas connection string |
| `frontend/.env.local` | `OPENROUTER_API_KEY` | OpenRouter API key |
| `frontend/.env.local` | `BACKEND_URL` | FastAPI backend URL (default: `http://localhost:8000`) |
| `frontend/.env.local` | `NEXT_PUBLIC_BACKEND_URL` | Client-side backend URL (same as above) |

## 👥 Authors

- **Mohammed Nassiri** — Frontend & LLM Integration
- **Yassine Esserdaoui** — Backend & Data Pipeline
