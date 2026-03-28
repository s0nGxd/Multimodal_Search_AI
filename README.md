# Multimodal Search Engine

An intelligent multimodal image search engine powered by **SigLIP** (vision-language model), **LanceDB** (vector store), and **BM25** keyword search — fused with Reciprocal Rank Fusion (RRF) for best-in-class retrieval accuracy.

---

## Prerequisites

Install these once on your machine:

- **Python 3.10+** — https://python.org
- **Node.js LTS** — https://nodejs.org

---

## Setup (First Time Only)

### Backend

**Windows (PowerShell)**
```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Mac / Linux**
```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd client
npm install
```

---

## Running the App

You need **two terminals** open simultaneously.

### Terminal 1 — Backend

**Windows**
```powershell
cd server
.\.venv\Scripts\Activate.ps1
python -m app.main
```

**Mac / Linux**
```bash
cd server
source .venv/bin/activate
python -m app.main
```

> ⏳ First run downloads the SigLIP model (~600 MB). Subsequent runs are fast.

Backend runs at: **http://localhost:8000**

### Terminal 2 — Frontend

```bash
cd client
npm run dev
```

Frontend runs at: **http://localhost:3000**

---

## Using the App

| URL | Purpose |
|---|---|
| http://localhost:3000 | Search images by text |
| http://localhost:3000/admin | Upload & index images (password: `admin`) |
| http://localhost:8000/docs | API documentation (Swagger UI) |

**First time?** Go to `/admin`, upload some images, then search at `/`.

---

## Architecture

```
client/          → Next.js frontend (search UI + admin upload panel)
server/
  app/
    main.py                        → FastAPI entry point
    routers/
      search.py                    → Search API endpoint
      upload.py                    → Image upload/ingestion endpoint
    services/
      search_service.py            → SigLIP embeddings + 3-channel RRF hybrid search
      ingestion_service.py         → Image download, captioning, embedding, LanceDB insert
      caption_service.py           → BLIP image captioning
      persistence_service.py       → HuggingFace dataset sync (for cloud deployment)
  .env                             → Model configuration (EMBED_MODEL, EMBED_DIM)
  requirements.txt                 → Python dependencies
```

## Search Strategy

Each search query runs three channels in parallel and fuses them with **Reciprocal Rank Fusion**:

1. **Image vector** — SigLIP visual similarity
2. **Caption vector** — SigLIP semantic text match against BLIP-generated captions
3. **BM25 keyword** — exact keyword match on image descriptions

The `% Match` score shown in the UI reflects how highly an image ranked across **all three channels combined** — not just geometric distance.
