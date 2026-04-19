# Changes for Final Report

This document tracks the key evolution points of the project to be included in the final report, specifically focusing on the transition from the interim submission to the final deliverable.

## 1. User Interface (UI) Overhaul
*Current Status: "Next-Gen Semantic Vision"*

We have completely redesigned the frontend to move away from a basic prototype look to a professional, "Dark Mode" aesthetic that emphasizes the AI nature of the product.

### Key Visual Changes:
- **Dark Mode Standard**: The application now defaults to a high-contrast dark theme (`bg-black`, `text-white`), replacing standard light/gray interfaces.
- **Glassmorphism**: Usage of translucent backdrops (`bg-white/5`, `backdrop-blur-xl`) for search bars and cards to create depth.
- **Gradient Typography**: Headlines use gradients (`bg-gradient-to-b from-white to-gray-500`) to feel modern and premium.
- **Ambient Lighting**: Background contains animated, blurred color orbs (`bg-purple-900/10`, `blur-[120px]`) to give a "glowing" AI feel without clutter.
- **Micro-Interactions**:
    - **Hover States**: Result images scale up (`scale-110`) and reveal metadata overlays on hover.
    - **Smooth Transitions**: Elements fade in using `framer-motion` (`animate={{ opacity: 1, y: 0 }}`).
    - **Loading States**: Replaced static text with animated spinners (`Loader2`).

---

## 2. Division of Labor & Role Evolution

The project work split remains as defined in the interim stage, but the **scope** of individual roles has evolved to meet new technical requirements.

### Original Split (Interim):
- **System 1 (Karl & Song)**: Data Collection & Embedding Generation.
- **System 2 (Bessie & Zheng)**: Indexing & Retrieval (Offline/Local).
- **System 3 (Aariz Sajan - User)**: UI & Integration.

### Evolved Scope (Final):
- **System 3 (Aariz Sajan) - Expanded Role**:
    - **From**: Purely UI and connecting to the local backend.
    - **To**: **Cloud Architecture & Online Indexing**.
    - **Reasoning**: The original "Offline-First" nature of System 2 (Bessie/Zheng) created a bottleneck where images had to be manually processed via CLI variables on a local machine.
    - **Implementation**:
        - Developed `IngestionService` to handle **Online Indexing**.
        - Moving database from local filesystem to **Cloud-Hosted/Persistent Storage** (e.g., separating the compute from the storage).
        - This allows the application to ingest images via URL/Upload in real-time, effectively "bringing the system online."

---

## 3. Strategic Evolution: Interim vs. Final Comparison

The following table summarizes the key deviations and improvements from the original Interim Report plan, demonstrating significant technical advancement.

| Category | **Interim Status (Dec 2025)** | **Final Status (Feb 2026)** | **Rationale for Evolution** |
| :--- | :--- | :--- | :--- |
| **Architecture** | **Offline-First (CLI)**. Images processed manually via local scripts. | **Online / Cloud-Ready**. Full REST API (`/api/ingest`) allowing real-time ingestion. | **User Experience**. Removes the need for admin technical intervention, transforming the tool into a self-serve CMS. |
| **Frontend Tech** | **Streamlit**. Python-based, rigid layout, limited interactivity. | **Next.js & React**. Commercial-grade framework with custom state management. | **Control**. Streamlit prevented custom "Match Score" overlays and micro-interactions essential for the "Next-Gen Vision". |
| **Backend Tech** | **Monolithic**. Tightly coupled with local filesystem. | **Microservice-Ready (FastAPI)**. Async architecture supporting background tasks (BLIP). | **Scalability**. Decoupling ingestion from search prevents blocking the main thread during heavy AI tasks. |
| **Models** | **Single Model (CLIP)**. Vector search only. | **Multi-Model (CLIP + BLIP)**. Vectors + Auto-Captioning. | **Data Richness**. Pure vectors are opaque; adding captions provides human-readable context and accessibility. |
| **Database** | **Local LanceDB (Ephemeral)**. File-based. | **Persistent Cloud DB**. Structured for Railway/Render persistence. | **Reliability**. Solves the "Amnesia Problem" where data is lost on server restarts. |

---

## 4. Final Polish & Advanced Controls (Implemented Feb 13, 2026)

To meet the "Customisation" requirement that was flagged as a risk in the Interim Report, the following granular controls have been added to the Frontend:

### 🎛️ Advanced Search Options
Users can now fine-tune the retrieval mechanism via a collapsible settings panel:
1.  **Max Results (K)**: Slider (1-50). Allows users to broaden or narrow the search scope.
2.  **Minimum Similarity (%)**: Slider (0% - 100%).
    - **Logic:** The backend uses *Cosine Distance* (where 0 is identical).
    - **Frontend Transformation:** We convert the user's "Similarity" preference into a "Distance Threshold" for the backend using the formula: `Distance_Threshold = 1 - (Similarity_Percentage / 100)`.
    - **Example:** A 90% strictness setting sends a distance threshold of `0.1`, filtering out any vaguely related matches.

### 🚀 Deployment Readiness
- **Code Status:** Feature Complete.
- **Next Step:** Deploy to **Railway** or **Render** with a Persistent Volume to ensure the LanceDB index survives server restarts (solving the "Online Amnesia" issue).

---

## 5. Considering Options To Take The Project Online

The application runs perfectly locally but has no deployment configuration. To go "online", we need to:
1. Deploy the **FastAPI backend** (with heavy ML models: CLIP ~600MB, BLIP ~1GB, PyTorch ~2GB) to a server with persistent storage.
2. Deploy the **Next.js frontend** to a CDN/edge platform.
3. Wire them together with environment variables (replacing hardcoded `localhost`).

> **Key Constraint:** PyTorch + Transformers require ~2-4GB RAM minimum. Free tiers on most platforms (Vercel, Render free, Railway free) will not work for the backend. A paid tier with ≥4GB RAM is needed.

### Platform Options Under Consideration

| Platform | Cost | RAM | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Render (Starter)** | $7/mo | 2GB | Simple setup, persistent disk | May be tight for CLIP+BLIP |
| **Render (Standard)** | $25/mo | 4GB | Comfortable headroom | Higher cost |
| **Railway** | ~$5-15/mo (usage) | Flexible | Pay-per-use, volume mounts | Less predictable billing |
| **Hugging Face Spaces** | Free | Varies | Free GPU/CPU for ML, no cost | Cold-start delays (~30s), less control |

### Backend Changes Required
- **Dockerfile**: Python 3.11 slim base, install deps, expose port 8000, run uvicorn.
- **CORS Configuration**: Make origins configurable via `ALLOWED_ORIGINS` env var (currently hardcoded to `localhost:3000`).
- **Port Configuration**: Read `PORT` from env (some platforms assign dynamically).
- **Dependencies**: Pin `torch` to CPU-only variant to reduce image size (~700MB instead of ~2GB).

### Frontend Changes Required
- **API Base URL**: Replace hardcoded `http://localhost:8000/api` with `process.env.NEXT_PUBLIC_API_URL`.
- **Image Domains**: Add remote patterns in `next.config.ts` to allow loading from deployed backend.
- **Production Env**: New `.env.production` file with `NEXT_PUBLIC_API_URL` pointing to deployed backend.

### Persistent Storage Strategy
- **LanceDB Data** (`lancedb_db/`): Mount a persistent disk/volume at `/data/lancedb_db` on the chosen platform.
- **Uploaded Images** (`data/`): Mount to the same volume at `/data/images`. Update static files path in `main.py` to be env-configurable.

### Verification Checklist (When Deployed)
1. `docker build -t segp-backend ./server` — must succeed
2. `curl https://<backend-url>/health` — must return `{"status":"ok"}`
3. `cd client && npm run build` — must succeed with new env var
4. Upload an image via `/admin` → search for it → verify result appears
5. Trigger backend restart → search again → image should persist (proves storage works)

### Decision: Hugging Face Spaces (Free Tier)

After careful consideration of the fact that the client has asked us to only create an MVP — or a "proof of concept" as he called it — we do not need to spend money on paid hosting. **Hugging Face Spaces** is the right choice because:
- It is **free** for community Spaces.
- We are still using **CLIP** (the VLM specified by the client) and the **open-source version of LanceDB** — so we are fully satisfying the tech stack requirements set by the client.
- The cold-start delay (~30s) is acceptable for an MVP/demo context.

### Action Plan: Deploying to Hugging Face Spaces + Vercel

#### Backend (FastAPI → Hugging Face Spaces)
1. **Create a `Dockerfile`** in `server/` — HF Spaces supports Docker-based apps.
2. **Add a `README.md`** with HF Spaces metadata (this tells HF it's a Docker Space, e.g. `sdk: docker`).
3. **Push to a Hugging Face repo** → it auto-builds and deploys.
4. **Persistent storage**: HF Spaces offers a `/data` persistent directory (free for community Spaces, but data resets if the Space sleeps for >48h on the free tier).

#### Frontend (Next.js → Vercel)
1. **Host on Vercel** — ideal for Next.js, free tier is more than sufficient for the frontend.
2. Set `NEXT_PUBLIC_API_URL` to the Hugging Face Space URL.
3. Update `next.config.ts` with the HF domain for remote image loading.

---

### 🚀 Milestone: Backend Successfully Deployed Online

The FastAPI backend is now **live** at: `https://aariz-s-segp-semantic-search.hf.space`

This is a significant transition — the project has moved from a local-only prototype to a publicly accessible, cloud-hosted API. Below is a summary of every file that was added or modified to achieve this.

#### New Files

| File | Purpose |
| :--- | :--- |
| **`server/Dockerfile`** | Docker build config for HF Spaces. Uses Python 3.11-slim, installs CPU-only PyTorch (saves ~1.3GB), creates persistent `/data` directories, and exposes port 7860. |
| **`server/README.md`** | HF Spaces metadata file. Specifies `sdk: docker`, `app_port: 7860`, and provides a brief overview of the API for the Space landing page. |

#### Modified Files

| File | What Changed | Why |
| :--- | :--- | :--- |
| **`server/app/main.py`** | CORS origins now read from `ALLOWED_ORIGINS` env var (was hardcoded `localhost:3000`). Data directory reads from `DATA_DIR` env var (was hardcoded `"data"`). Port reads from `PORT` env var (was hardcoded `8000`). Added a root route `GET /` returning API info. | Allows the app to work both locally and when deployed. The root route fixes HF Spaces' health check, which pings `/` and expected a 200 response. |
| **`server/app/services/ingestion_service.py`** | `self.data_dir` reads from `DATA_DIR` env var. `photo_url` uses `BACKEND_URL` env var (was hardcoded `http://localhost:8000`). | Uploaded images must be accessible via the deployed URL, not localhost. Data must write to HF's persistent `/data` directory. |
| **`client/lib/api.ts`** | `API_BASE` reads from `NEXT_PUBLIC_API_URL` env var (was hardcoded `http://localhost:8000/api`). | The frontend needs to point to the deployed backend URL when hosted on Vercel. |
| **`server/requirements.txt`** | Removed `streamlit` (legacy, unused). Added `pandas` and `requests` (were imported but unlisted). Reorganised with category comments. | Reduces Docker image size and prevents build failures from missing dependencies. |

#### Key Design Decisions
- **CPU-only PyTorch**: Saves ~1.3GB in Docker image size, critical for free-tier resource limits.
- **Environment variables for all config**: `ALLOWED_ORIGINS`, `BACKEND_URL`, `DATA_DIR`, `LANCEDB_URI`, `PORT`, `NEXT_PUBLIC_API_URL` — makes the app deployment-agnostic.
- **HF Spaces persistent `/data`**: LanceDB index and uploaded images survive container restarts.
- **Root route added**: HF Spaces pings `GET /` for health checks; without it, the Space showed as broken despite the server running fine.

#### 🌐 Frontend Goes Live — Connecting the Two Halves

With the backend running, the next challenge was getting the frontend online and making the two halves talk to each other.

**Vercel** was the natural choice for hosting a Next.js app — it's built by the same team, and the free tier is more than sufficient. Deploying was straightforward via the Vercel CLI (`vercel --prod`), but the real lessons came from what happened *after* the first deploy:

1. **`NEXT_PUBLIC_*` variables are baked at build time.** Unlike server-side environment variables, Next.js inlines `NEXT_PUBLIC_*` values into the JavaScript bundle during `next build`. This means setting the variable in Vercel's project settings isn't enough — you have to **redeploy** for the change to take effect. The first deployment used `localhost:8000` because the env var wasn't yet configured; a second build was required after adding it to Vercel's environment settings.

2. **CORS — the inevitable cross-origin wall.** The frontend at `client-nine-xi-31.vercel.app` and the backend at `aariz-s-segp-semantic-search.hf.space` are on completely different domains. The browser's same-origin policy blocks all API requests unless the backend explicitly allows the frontend's origin in its CORS headers. The fix was adding the Vercel domain to the `default_origins` in `main.py` and pushing the update to HF Spaces. This is a classic deployment pitfall — everything works locally because both frontend and backend share `localhost`, and the cross-origin issue only surfaces once they're on separate domains.

3. **Remote images require explicit allowlisting in Next.js.** The `<Image>` component in Next.js refuses to load images from external domains by default (a security measure). The `next.config.ts` had to be updated with `remotePatterns` for the HF Spaces hostname — otherwise, every search result image would fail to render.

4. **The ghost localhost in the database.** After the first end-to-end test, the search returned the correct result — but the image was a broken icon. The search *worked* (CLIP found the right match), proving the ML pipeline was functional. But the image URL stored in LanceDB was `http://localhost:8000/images/...` because `BACKEND_URL` had been left out of the Dockerfile. The ingestion service defaulted to localhost when it wrote the record. The subtle part: fixing the Dockerfile and redeploying doesn't fix existing data. The wrong URL is already baked into the LanceDB record. The image had to be **re-uploaded** after the fix — a reminder that misconfigured writes to a database persist long after the code is corrected.

**Result:** The full application — frontend on Vercel, backend on HF Spaces — is now publicly accessible. Each deployment issue surfaced only in production, reinforcing that local development masks an entire class of cross-domain, environment, and configuration problems that only appear when backend and frontend live on separate infrastructure.

| Service | URL |
| :--- | :--- |
| **Frontend** | `https://client-nine-xi-31.vercel.app` |
| **Backend API** | `https://aariz-s-segp-semantic-search.hf.space` |

---

## 6. Solving Persistent Storage on HF Spaces Free Tier (Testing)

### The Problem
HF Spaces free tier has **ephemeral storage**. Container restarts wipe the LanceDB index and uploaded images — meaning any data ingested via the admin panel is lost when the Space sleeps or redeploys.

### The Proposed Solution: HF Dataset Repository as a Persistence Layer
Use a **Hugging Face Dataset repository** as a free, Git LFS-backed persistence layer. The data gets synced to/from a dedicated repo automatically.

### How It Works
1. **On startup**: Download any existing data (LanceDB files + images) from the HF Dataset repo to `/data`.
2. **After each upload**: Sync the updated data back to the repo (async, non-blocking).
3. **On next restart**: Step 1 restores everything automatically.

```
Container starts → download from HF repo → data restored → app runs normally
User uploads image → save locally + embed + index → sync to HF repo (background)
Container dies → data lost locally → BUT repo still has it
Container restarts → download from repo → fully restored
```

> **Requirement:** A HF write token must be set as `HF_TOKEN` secret in the Space settings.

### Files Changed / Created

| File | Action | Purpose |
| :--- | :--- | :--- |
| `server/app/services/persistence_service.py` | **NEW** | `download_from_repo()` — called on startup, pulls LanceDB + images from HF Dataset repo. `upload_to_repo()` — called after ingestion, pushes updated files back in a background thread. Uses `huggingface_hub` library. |
| `server/app/main.py` | **MODIFY** | Add `@app.on_event("startup")` handler that calls `persistence_service.download_from_repo()` to restore data before any search requests come in. |
| `server/app/services/ingestion_service.py` | **MODIFY** | After each successful ingestion (`process_upload`, `process_url_upload`), trigger a background sync to the HF Dataset repo. |
| `server/Dockerfile` | **MODIFY** | Add `huggingface_hub` to installed packages. Add `HF_DATASET_REPO` env var. |
| `server/requirements.txt` | **MODIFY** | Add `huggingface-hub` dependency. |

### Setup Steps
1. Create a new HF Dataset repo (e.g. `aariz-s/segp-search-data`).
2. Add `HF_TOKEN` as a secret in the HF Space settings (Settings → Variables and secrets).
3. Push the updated code.

### Verification: Confirmed Working

Persistence was tested on **17 Feb 2025** with the following procedure:

1. Three images were uploaded via the Admin panel (a black car photo and two AI-generated images).
2. A search for *"a close up of a black car"* returned all three results with correct similarity scores.
3. The HF Dataset repo (`aariz-s/segp-search-data`) was checked — both the `images/` and `lancedb_db/` folders had been synced automatically within minutes.
4. A container restart was triggered by pushing a dummy commit to the Space.
5. After the full rebuild (~3 minutes), the same search was repeated — **identical results were returned**, proving the data survived the restart.

### Evidence Screenshots

The following screenshots document the persistence layer in action. Each one shows a different aspect of the HF Dataset repository that serves as the application's cloud backup.

#### Screenshot 1: Dataset Viewer — The Persisted Data at a Glance

This screenshot shows the Hugging Face Dataset Viewer, which automatically parses the stored LanceDB data into a browsable table. It confirms that 3 records are persisted, each containing:
- **`photo_id`** — the unique identifier for the image
- **`photo_image_url`** — the full URL pointing back to the HF Space's image endpoint
- **`description`** — the AI-generated caption from BLIP (e.g. *"a black car parked on the street"*, *"a man with a beard and a white sweater"*)
- **`vector`** — the 512-dimensional CLIP embedding used for semantic search

This view proves that the search index isn't just a collection of files — it's structured, queryable data that Hugging Face can interpret as a real dataset.

#### Screenshot 2: Repository Structure — Where the Data Lives

This screenshot shows the "Files and versions" tab of the Dataset repo. The two key directories are:
- **`images/`** — contains the actual uploaded image files (the binary data)
- **`lancedb_db/`** — contains the LanceDB vector database files (the search index)

The commit messages (*"Sync uploaded images"*, *"Sync LanceDB index"*) confirm that these were pushed automatically by the persistence service's background thread, not manually uploaded. The "VERIFIED" badge indicates the commits were authenticated via the `HF_TOKEN` secret.

#### Screenshot 3: Images Folder — The Uploaded Files

This screenshot shows the contents of the `images/` directory. Three files are stored:
- `_.jpeg` (82.4 kB) — the black car photo
- `Gemini_Generated_Image_ihj6ho...png` (7.26 MB) — an AI-generated test image
- `Gemini_Generated_Image_bulh0h...png` (7.54 MB) — another AI-generated test image

These are stored using Git LFS (Large File Storage), indicated by the `xet` badges. This is significant because Git normally struggles with binary files — LFS handles them efficiently by storing only pointers in the Git history and the actual binary content separately.

#### Screenshot 4: LanceDB Index Files — The Search Brain

This screenshot shows the internal structure of the LanceDB database at `lancedb_db/images.lance/data/`. The `.lance` files are the actual vector index fragments — each one contains the CLIP embeddings and metadata for the stored images. These files are what makes semantic search possible: when a user types a query, CLIP converts it to a vector, and LanceDB searches these files to find the closest matches.

The fact that these files are synced to the Dataset repo means the entire search capability — not just the images — persists across container restarts.

---

## 7. Performance Optimization Sprint (March 26, 2026)

The system was functionally complete but had severe performance bottlenecks that made it impractical beyond a quick demo. Six targeted optimizations were applied to the backend without changing the frontend UI, the LanceDB schema, or the CLIP model.

### Changes Made

**1. Lazy-Loaded BLIP Model**
- CaptionService no longer loads the ~1GB BLIP model at import time.
- Model loads on first `generate_caption()` call only.
- Search-only sessions (99% of traffic) skip BLIP entirely.
- Cold start reduced from ~60s to ~30s.
- File: `server/app/services/caption_service.py`

**2. LRU Cache on Text Embeddings**
- Added an OrderedDict-based LRU cache (128 entries) to `embed_text()`.
- Identical/repeated queries return instantly from cache.
- Cache key is normalized (lowercased + stripped) for better hit rate.
- Max memory overhead: ~256KB.
- File: `server/app/services/search_service.py`

**3. Batch CLIP Embeddings**
- New `embed_images_batch()` method processes up to 16 images per CLIP forward pass.
- Replaces one-at-a-time inference in bulk ingestion.
- Exploits internal parallelism of the transformer model.
- File: `server/app/services/search_service.py`

**4. Concurrent Image Downloads**
- Bulk CSV ingestion now uses `ThreadPoolExecutor(max_workers=8)` for parallel downloads.
- Downloads are network-bound, so threading provides near-linear speedup.
- Image download function extracted as a standalone helper for clean error handling per image.
- File: `server/app/services/ingestion_service.py`

**5. Optional Captioning in Bulk Mode**
- New `generate_captions` parameter (default: `False`) on `process_bulk_csv()`.
- Single uploads and URL ingestion still generate BLIP captions (important for individual context).
- Bulk mode skips captioning by default since search uses CLIP vectors, not text descriptions.
- Saves ~200ms per image (3+ minutes on 1000 images).
- Files: `server/app/services/ingestion_service.py`, `server/app/routers/upload.py`

**6. IVF-PQ Index Auto-Creation**
- After bulk ingestion, if table has >= 256 rows, an IVF-PQ index is automatically built.
- `num_partitions = sqrt(row_count)`, `num_sub_vectors = 16`.
- Converts brute-force O(n) search to approximate nearest neighbor O(log n).
- Index is rebuilt with `replace=True` so it stays current.
- File: `server/app/services/ingestion_service.py`

### Files Modified

| File | What Changed |
| :--- | :--- |
| `server/app/services/caption_service.py` | Replaced eager initialization with lazy `_ensure_loaded()` pattern |
| `server/app/services/search_service.py` | Added LRU text cache + `embed_images_batch()` method |
| `server/app/services/ingestion_service.py` | Concurrent downloads, batch embeddings, optional captions, IVF-PQ index |
| `server/app/routers/upload.py` | Added `generate_captions` field to `BulkIngestRequest` |

### Performance Impact Summary

| Metric | Before | After |
| :--- | :--- | :--- |
| Cold start (search-only) | ~60s (CLIP + BLIP) | ~30s (CLIP only) |
| Repeated query latency | ~200ms (full CLIP inference) | <1ms (cache hit) |
| Bulk ingestion (100 images) | ~5-10 min (sequential) | ~1-2 min (concurrent + batched) |
| Search at 10k+ images | O(n) brute-force | O(log n) with IVF-PQ |

---

## 8. Hybrid Search — Dual-Vector Architecture (March 26, 2026)

A fundamental limitation was discovered in the search accuracy: the system generates BLIP captions for every image, but searching for the exact caption only produced ~50-60% match scores. This is because CLIP's text encoder and image encoder produce embeddings in related but not identical spaces — cross-modal similarity is inherently lower than same-modal similarity.

### The Fix: Store Two Vectors Per Image

At ingestion time, we now store two CLIP vectors per image:
1. **Image vector** — from CLIP's vision encoder (existing)
2. **Caption vector** — from CLIP's text encoder, encoding the BLIP-generated caption (new)

At search time, the user's query is compared against both vectors. For each image, the better score wins.

### How It Works

```
User query: "elephants walking on a road"
    ↓
CLIP text encoder → query vector
    ↓
Search 1: query vector vs image vectors (visual match)    → 81%
Search 2: query vector vs caption vectors (text match)     → 95%
    ↓
Merge: take best score per image → 95%
```

For exact BLIP captions, the caption vector search produces 100% (identical text through the same encoder = identical vectors).

### Files Modified

| File | What Changed |
| :--- | :--- |
| `server/app/services/ingestion_service.py` | Added `caption_vector` field to `ImageRecord` schema. After BLIP captioning, caption is encoded via CLIP text encoder and stored alongside image vector. |
| `server/app/services/search_service.py` | `search()` now runs two vector searches (image + caption columns), merges by taking the best distance per `photo_id`, sorts, and returns top-k. |

### Performance Impact

| Metric | Before | After |
| :--- | :--- | :--- |
| Exact caption search score | ~50-60% | **100%** |
| Paraphrased caption search | ~40-50% | **70-90%** |
| Visual queries (e.g. "elephant crossing") | 50-81% | **81-100%** |
| Storage per image | ~2KB (1 vector) | ~4KB (2 vectors) |
| Search latency | 1 LanceDB query | 2 LanceDB queries + merge |

The storage doubling is negligible at this dataset scale. The extra search query adds minimal latency since both queries hit the same in-memory table.

### Design Rationale

This approach was chosen over alternatives because:
- **No model changes** — still uses CLIP and BLIP, just wires them together more intelligently
- **No schema migration** — the table is recreated on re-ingestion (acceptable for the project's dataset size)
- **Graceful fallback** — if caption_vector is all zeros (e.g., bulk ingestion without captions), the image vector search still works normally

---

## 9. Score Rescaling Recalibration (March 26, 2026)

Immediately after deploying hybrid search (Section 8), all results displayed 100% Match — the rescaling constants from the earlier calibration (Section 7) were invalidated by the wider score range of caption vector matches.

### What Changed

The rescaling floor/ceiling were updated from image-only range to hybrid search range:

| Constant | Before (image-only) | After (hybrid) |
| :--- | :--- | :--- |
| `CLIP_SIM_FLOOR` | 0.10 | **0.40** |
| `CLIP_SIM_CEIL` | 0.35 | **1.00** |

### Files Modified

| File | What Changed |
| :--- | :--- |
| `server/app/routers/search.py` | Updated `CLIP_SIM_FLOOR` to 0.40, `CLIP_SIM_CEIL` to 1.00 |
| `client/app/page.tsx` | Updated frontend threshold constants to match backend range |

### Result

| Query | Top match score (before fix) | Top match score (after fix) |
| :--- | :--- | :--- |
| "elephant crossing" → elephant image | 100% (all results 100%) | **66%** (others 30-42%) |
| Exact BLIP caption → source image | 100% | **100%** (others 26-32%) |

Scores now properly differentiate matches from non-matches. Exact caption searches still reach 100%.

### Lesson

Each layer of search improvement (image vectors → rescaling → hybrid search) shifted the raw score distribution. Rescaling constants are not one-time values — they must be recalibrated whenever the underlying scoring mechanism changes. This is a general principle: any normalization layer must be validated end-to-end after upstream changes.

