# Developer Setup Guide

Step-by-step instructions to run the SEGP Semantic Search Engine locally and deploy to production.

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- npm
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/wysetime-collaboration/unm-segp-group-20-autumn-2025.git
cd unm-segp-group-20-autumn-2025
```

---

## 2. Backend Setup (FastAPI)

```bash
cd server

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install PyTorch (CPU-only, smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### Configure Environment Variables

Create `server/.env`:

```env
LANCEDB_URI=./lancedb_db
EMBED_MODEL=openai/clip-vit-base-patch32
```

Optional (for HF Hub persistence):

```env
HF_DATASET_REPO=aariz-s/segp-search-data
HF_TOKEN=your_huggingface_token
BACKEND_URL=http://localhost:8000
```

### Run the Backend

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

First startup takes ~30s as CLIP and BLIP models are downloaded and loaded into memory.

---

## 3. Frontend Setup (Next.js)

```bash
cd client

# Install dependencies
npm install
```

### Configure Environment Variables

For local development, the frontend defaults to `http://localhost:8000/api`. No configuration needed.

For production, `client/.env.production` is already set:

```env
NEXT_PUBLIC_API_URL=https://aariz-s-segp-semantic-search.hf.space/api
```

### Run the Frontend

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## 4. Test the System

1. Open `http://localhost:3000/admin`, password: `admin`
2. Upload an image via **File Upload**
3. Go back to `http://localhost:3000`
4. Search for something that describes your image
5. Verify the image appears in results

---

## 5. Deploy to Production

### Backend — Hugging Face Spaces

The backend is deployed as a Docker container on HF Spaces.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Hardware: **CPU Basic** (free tier, 2 vCPU / 16GB RAM)

2. Push the `server/` directory contents to the Space repo:
   ```bash
   # From the server/ directory
   huggingface-cli upload <your-space-id> . . --repo-type space
   ```

3. Set Space secrets (Settings > Variables and secrets):
   - `HF_TOKEN` — Your HF access token (for persistence)
   - `HF_DATASET_REPO` — e.g., `aariz-s/segp-search-data`
   - `BACKEND_URL` — e.g., `https://<your-space>.hf.space`

4. The Space will build and deploy automatically.

### Frontend — Vercel

1. Import the GitHub repo on [vercel.com/new](https://vercel.com/new)
2. Set **Root Directory** to `client`
3. Set environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://<your-hf-space>.hf.space/api`
4. Deploy

Or via CLI:

```bash
cd client
vercel --prod
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/search` | Search images by text query |
| `GET` | `/api/images/all` | List all indexed images |
| `POST` | `/api/upload` | Upload and index a single image |
| `POST` | `/api/ingest/url` | Index an image from a URL |
| `POST` | `/api/ingest/bulk` | Bulk import from CSV dataset |
| `DELETE` | `/api/clear-db` | Clear the entire database |

### Search Request

```json
POST /api/search
{
  "query": "sunset over mountains",
  "k": 10,
  "threshold": 0.6
}
```

### Search Response

```json
[
  {
    "photo_id": "image-001",
    "photo_image_url": "https://.../image.jpg",
    "description": "a mountain range at sunset with orange sky",
    "score": 0.82
  }
]
```

---

## Troubleshooting

**Backend won't start / model download fails**
- Ensure you have ~3GB free disk space for CLIP + BLIP model downloads
- Check your internet connection — models are downloaded from Hugging Face Hub on first run

**Search returns empty results**
- Verify the backend is running: `curl http://localhost:8000/health`
- Check that you've ingested at least one image via the admin panel
- If the LanceDB directory is missing, ingest will create it automatically

**CORS errors in browser console**
- The backend allows `localhost:3000` and `*.vercel.app` by default
- For custom domains, set the `ALLOWED_ORIGINS` env var on the backend

**HF Spaces cold start**
- Free tier sleeps after ~15 minutes of inactivity
- First request after sleep takes ~60-80s (container boot + model loading)
- The frontend shows a "Waking up search engine" indicator during this time
