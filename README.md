# SEGP Semantic Search Engine

A multimodal semantic image search engine that lets users find images using natural language. Built for **WyseTime Technologies** as part of the Software Engineering Group Project (SEGP).

**Live Demo:** [https://client-nine-xi-31.vercel.app](https://client-nine-xi-31.vercel.app)

---

## What It Does

Type a description — "elderly man wearing glasses", "red trolley on a city street", "mountain landscape" — and the system returns matching images ranked by semantic similarity.

### Key Capabilities

- **Natural Language Search** — Search by concepts, not filenames. Powered by OpenAI's CLIP model.
- **Auto-Captioning** — Every uploaded image is automatically described using BLIP, enabling text-to-text matching alongside visual matching.
- **Hybrid Search** — Queries are matched against both the image embedding and the caption embedding. The best match wins.
- **Online Ingestion** — Upload images via the admin panel (file upload, URL, or bulk CSV import). Images become searchable immediately.
- **Persistent Storage** — Data survives server restarts via automatic sync to Hugging Face Hub.

---

## Architecture

```
Next.js (Vercel)  →  FastAPI (HF Spaces)  →  LanceDB + HF Hub
   Frontend             Backend API           Vector DB / Persistence
```

| Component | Technology | Hosting |
|---|---|---|
| Frontend | Next.js 16, React 19, Tailwind CSS | Vercel |
| Backend API | FastAPI, Python 3.11 | Hugging Face Spaces (Docker) |
| Embedding Model | CLIP ViT-B/32 (512-dim vectors) | Loaded at runtime |
| Captioning Model | BLIP (Salesforce) | Loaded at runtime |
| Vector Database | LanceDB (open source) | Local + HF Hub sync |

---

## How It Works

### Search Flow
1. User enters a text query
2. CLIP encodes the query into a 512-dimensional vector
3. LanceDB searches against both image vectors and caption vectors
4. Results are ranked by cosine similarity and returned with match scores

### Ingestion Flow
1. Image is uploaded via the admin panel
2. CLIP generates an image embedding (512-dim vector)
3. BLIP generates a text caption
4. CLIP embeds the caption into a second vector
5. Both vectors + metadata are stored in LanceDB
6. Data is synced to Hugging Face Hub for persistence

---

## Using the System

### Search (Public)
1. Open [https://client-nine-xi-31.vercel.app](https://client-nine-xi-31.vercel.app)
2. Wait for the green "Search engine ready" indicator
3. Type a description and click Search
4. Click any image to view it in full size with its caption

### Admin Panel
1. Navigate to [/admin](https://client-nine-xi-31.vercel.app/admin)
2. Password: `admin`
3. Available actions:
   - **File Upload** — Upload a single image from your device
   - **Remote URL** — Index an image from any public URL
   - **Bulk Ingest** — Import images from the reference dataset
   - **All Images** — View every indexed image in the database

---

## Performance

| Metric | Value |
|---|---|
| Search response time (warm) | ~1.1s |
| Single image ingestion | ~13s |
| Bulk ingestion throughput | ~24 images/min |
| Cold start (after sleep) | ~79s |

### Search Accuracy (15 test queries, 30 images)

| Metric | Score |
|---|---|
| Precision@1 | 73.3% |
| Precision@3 | 80.0% |
| Precision@10 | 86.7% |
| Mean Reciprocal Rank | 0.84 |

Full evaluation details in [`DOCUMENTATION/evaluation-metrics.md`](DOCUMENTATION/evaluation-metrics.md).

---

## Project Structure

```
├── README.md                  ← You are here
├── SETUP.md                   ← Developer setup guide
├── DOCUMENTATION/
│   ├── evaluation-metrics.md  ← Search accuracy benchmarks
│   ├── architecture_diagrams.md
│   └── screenshots/
├── client/                    ← Next.js frontend
│   ├── app/                   ← Pages (search, admin)
│   ├── lib/api.ts             ← API client
│   └── .env.production        ← Production API URL
└── server/                    ← FastAPI backend
    ├── app/
    │   ├── main.py            ← FastAPI app entry point
    │   ├── routers/           ← API endpoints
    │   └── services/          ← CLIP, BLIP, LanceDB, persistence
    ├── requirements.txt
    └── Dockerfile
```

---

## Team

Built by **Group 20** — University of Nottingham Malaysia, Software Engineering Group Project (Autumn 2025).

Client: **WyseTime Technologies Sdn. Bhd.** — AI computer vision solutions for security and surveillance.
