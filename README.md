# Multimodal Search Engine

An intelligent multimodal image and video search engine powered by **SigLIP** (vision-language model) and **LanceDB** (vector store), with **OWL-ViT** object detection re-ranking and an optional **Gemini** parser + VLM re-rank for compositional and action-aware queries.

---

## Prerequisites

Install these once on your machine:

- **Python 3.10+** — https://python.org
- **Node.js LTS** — https://nodejs.org

---

## Setup (First Time Only)

### Environment files

Before installing dependencies, set up the env files:

```bash
# Backend
cp server/.env.example server/.env

# Frontend (only needed if you want the Gemini parser + VLM features locally)
# Create client/.env.local with:
#   GOOGLE_PROJECT_ID=...
#   GOOGLE_CLIENT_EMAIL=...
#   GOOGLE_PRIVATE_KEY=...
# (ask Aariz for the values, or skip this and the app falls back to basic search)
```

### Backend

**Windows (PowerShell)**
```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# OPTION A: Standard (CPU Only)
pip install -r requirements.txt

# OPTION B: GPU Accelerated (Recommended for Nvidia users)
# Run this AFTER Option A to upgrade to the CUDA-enabled engine:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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
python -m uvicorn app.main:app --reload --port 8000
```

**Mac / Linux**
```bash
cd server
source .venv/bin/activate
python3 -m uvicorn app.main:app --reload --port 8000
```

> ⚠️ Use `python3 -m uvicorn`, not a venv-activated `uvicorn`. The project venv may have `transformers 5.x` which breaks SigLIP — system Python (4.x) is confirmed working. See gotchas below.
>
> ⏳ First run downloads SigLIP (~780 MB) and OWL-ViT (~590 MB). Subsequent runs load from `~/.cache/huggingface/hub/` and are much faster (~30s on CPU).

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

**First time?** Either go to `/admin` and upload some images, or restore the production dataset locally (see next section) so you can search the same data the live demo uses.

---

## Run against prod data locally (read-only)

By default the local backend serves whatever is in `server/lancedb_db/` and `server/data/`. To test against the same images and videos the live demo sees, restore the HuggingFace dataset on boot — no risk of polluting prod, the sync path is gated off.

1. Get a HuggingFace **read-scope** token at https://huggingface.co/settings/tokens
2. Add to `server/.env`:
   ```
   HF_DATASET_REPO=aariz-s/segp-siglip-data
   HF_TOKEN=hf_...
   PERSISTENCE_READ_ONLY=1
   ```
3. Restart uvicorn. First boot downloads the dataset (a few minutes); subsequent boots are fast.

You should see this on startup:
```
[Persistence] Restoring data from repo: aariz-s/segp-siglip-data (READ-ONLY)
[Persistence] Restore complete.
```

`PERSISTENCE_READ_ONLY=1` turns `sync_to_repo()` into a no-op, so local uploads and ingestion never push back to the live dataset. Remove the flag only if you actually want your local changes to sync upstream.

---

## Architecture

```
client/                              → Next.js App Router frontend
  app/
    page.tsx                         → Search UI
    admin/                           → Upload + ingestion UI (password: admin)
    api/
      parse-query/route.ts           → Gemini parser → compositional plan (OR/AND/NOT)
      vlm-rerank/route.ts            → Gemini VLM re-rank for action / attribute fidelity
  lib/
    api.ts                           → Backend client + readiness polling
    vertex.ts                        → Vertex AI auth
server/
  app/
    main.py                          → FastAPI entry, lifespan warmup, CORS
    routers/
      search.py                      → /api/search endpoint
      upload.py                      → /api/upload + ingestion endpoint
    services/
      search_service.py              → SigLIP text/image embed + LanceDB cosine search
      detection_service.py           → OWL-ViT detection re-rank
      ingestion_service.py           → Image / video frame embed + LanceDB insert
      persistence_service.py         → HF dataset restore (boot) + sync (after upload)
  .env                               → Local config (model, persistence, etc.)
  requirements.txt
```

## Search Strategy

The pipeline is a single semantic-search path with optional Gemini stages on either end:

1. **(Optional) Parse** — Gemini turns the natural-language query into a structured plan with sub-queries (`OR` / `AND` / `NOT`). Skipped if Vertex creds aren't set; the raw query is used directly.
2. **Embed** — SigLIP text encoder produces a 768-dim embedding for the query (or each sub-query).
3. **Vector search** — LanceDB cosine search returns the top-15 candidate images / video frames.
4. **OWL-ViT re-rank** — each candidate is scored against the query as a detection prompt; high-confidence detections lift the score into the 50–90% band. Threshold filtered.
5. **(Optional) VLM re-rank** — top results are graded by a Gemini multimodal model for action and attribute fidelity (e.g. "person *carrying* a backpack" vs "person standing next to a backpack"). Skipped if Vertex creds aren't set.

The `% Match` shown in the UI is the rescaled cross-modal similarity (or VLM-graded score, when VLM is on). With `SIM_FLOOR=0.00`, `SIM_CEIL=0.20` in [`search_service.py`](server/app/services/search_service.py), pure-SigLIP scores for relevant top results sit around 15–40%; OWL-ViT and VLM lift confident matches into the 50–90% band.

There is **no captioning channel** and **no BM25** — the BLIP caption + BM25 fusion was removed in favor of pure SigLIP + OWL-ViT detection re-rank, which gave more honest scores and fewer "wrong but confident" results.

---

## Access to deployed services

Production runs on Aariz's accounts. If you want push/deploy access, message Aariz with your username on the relevant platform and he'll add you:

- **HuggingFace Space (backend):** `aariz-s/segp-siglip-search` — send your HF username, get added with Write permission. Then you can `git push` to the Space's repo.
- **HuggingFace Dataset (LanceDB + indexed images):** `aariz-s/segp-siglip-data` — same drill. Send your HF username.
- **Vercel (frontend):** the project is on Aariz's personal Vercel team. Deployments are manual via `vercel --prod --yes`. Ask Aariz to deploy or to invite you.
- **Vertex AI / Gemini credentials:** ask Aariz for the `GOOGLE_*` env vars to enable the parser and VLM re-rank features locally.

Don't have an HF account? Sign up free at https://huggingface.co/join.

---

## Common gotchas

- **Backend loads CLIP instead of SigLIP.** Your `server/.env` has a stale `EMBED_MODEL=openai/clip-vit-base-patch32` from before the SigLIP migration. Fix:
  ```
  EMBED_MODEL=google/siglip-base-patch16-256
  EMBED_DIM=768
  ```
  If you've already ingested anything with the wrong model, hit `DELETE /api/clear-db` before re-ingesting — the dimension mismatch will silently break searches.

- **`ImportError: SiglipTokenizer requires the SentencePiece library`.** Your venv was created before SigLIP was added. Either `pip install -r requirements.txt` again, or just `pip install sentencepiece`.

- **`AttributeError: 'NoneType' object has no attribute 'replace'` when loading SigLIP processor.** The venv has `transformers 5.x` installed which breaks SigLIP's `AutoProcessor`. The project requires `transformers>=4.44.0,<5`. Fix: run `pip install "transformers>=4.44.0,<5"` inside the venv, or bypass the venv entirely and use system Python:
  ```bash
  # Mac/Linux — use system python3 instead of the venv
  cd server
  python3 -m uvicorn app.main:app --reload --port 8000
  ```

- **Always launch uvicorn from inside `server/`, not the project root.** `LANCEDB_URI=./lancedb_db` and `DATA_DIR=data` are relative paths. If uvicorn runs from the project root, restored data lands at `./lancedb_db` (project root) instead of `server/lancedb_db/`, and searches return 0 results. Always `cd server` first.

- **Frontend ends up on `:3001`.** Next.js falls back when `:3000` is busy. The CORS allowlist already includes both, but if the UI sits on "Backend not ready" indefinitely, open browser devtools → Network and check whether `/ready` is being CORS-blocked or returning an error.

- **Turbopack panics on `npm run dev` (Next 16.x).** A known bug in the default bundler manifests as `inner_of_upper_lost_followers ... panicked` in the dev server log. Workaround:
  ```bash
  rm -rf client/.next
  npx next dev --webpack
  ```

- **`[Persistence] Not configured` at startup.** Harmless if you only want to run against `server/lancedb_db/` and `server/data/` as-is. Only matters if you've set `HF_DATASET_REPO` + `HF_TOKEN` and *expected* a restore — in which case check that `load_dotenv()` is finding `server/.env` (it must be invoked from inside `server/`).

- **`/api/images/all` returns 0 after a successful restore.** Usually means `server/lancedb_db/` had stale files before the restore ran, and `_copy_tree` merged them with the restored files, producing a corrupt mixed state. LanceDB will list the table but refuse to open it. Fix:
  ```bash
  # Stop uvicorn first, then:
  rm -rf server/lancedb_db
  # Restart uvicorn — restore will write into a clean directory this time
  ```

- **`table_names()` returns `['images']` but `open_table('images')` raises `Table 'images' was not found`.** Corrupt LanceDB state — version manifests show numbers near `u64::MAX` (`18446744073709551587...`), meaning a prior `DELETE /api/clear-db` was synced to the HF dataset. The fix is the same as above: `rm -rf server/lancedb_db` and restart.

- **Gallery thumbnails show as broken images.** Usually means `photo_image_url` in LanceDB doesn't match the actual file on disk in `server/data/`. Often a fallout of an older ingestion run; clear and re-ingest.

---

## Performance & Security

### 🚀 GPU Acceleration
The application automatically detects Nvidia GPUs (`cuda`) and uses them as the primary compute engine. Transitioning from CPU to GPU typically results in a **10x-20x speedup** for both search and video indexing.

### 🛡️ Safetensors (CVE-2025-32434)
This engine utilizes the **Safetensors** weight format for all models (SigLIP and OWL-ViT). This protects the application from the well-known "Pickle vulnerability" (CVE-2025-32434) by ensuring that model weights are loaded in a restricted, non-executable memory space.
