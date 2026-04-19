# Architecture Diagrams

These diagrams are written in Mermaid syntax. To render them:
- Paste into https://mermaid.live for PNG/SVG export
- Or use a Mermaid-compatible Markdown viewer (VS Code with Mermaid extension, GitHub, etc.)

---

## 1. System Deployment Architecture

Shows where each component is hosted and how they communicate.

```mermaid
graph TB
    subgraph "User's Browser"
        USER[End User / Admin]
    end

    subgraph "Vercel (Frontend Hosting)"
        NEXT["Next.js + React<br/>client/"]
    end

    subgraph "Hugging Face Spaces (Backend Hosting)"
        subgraph "Docker Container"
            FASTAPI["FastAPI Server<br/>Port 7860"]
            CLIP["CLIP ViT-B/32<br/>~400MB<br/>Loaded at startup"]
            BLIP["BLIP Captioning<br/>~990MB<br/>Lazy-loaded on first caption"]
            LANCE["LanceDB<br/>(Embedded Vector DB)"]
            PERSIST["Persistence Service<br/>(Background Sync)"]
        end
        subgraph "Persistent Volume /data"
            IMAGES["/data/images/<br/>Uploaded image files"]
            LANCEDB["/data/lancedb_db/<br/>Vector index files"]
        end
    end

    subgraph "Hugging Face Hub"
        HFREPO["HF Dataset Repository<br/>aariz-s/segp-search-data<br/>(Git LFS backup)"]
    end

    USER -->|"HTTPS"| NEXT
    NEXT -->|"REST API calls<br/>NEXT_PUBLIC_API_URL"| FASTAPI
    FASTAPI --> CLIP
    FASTAPI --> BLIP
    FASTAPI --> LANCE
    LANCE --> LANCEDB
    FASTAPI --> IMAGES
    PERSIST -->|"Sync on write<br/>(background thread)"| HFREPO
    HFREPO -->|"Restore on startup"| LANCEDB
    HFREPO -->|"Restore on startup"| IMAGES

    style CLIP fill:#4a9eff,color:#fff
    style BLIP fill:#ff6b6b,color:#fff
    style LANCE fill:#51cf66,color:#fff
    style HFREPO fill:#ffd43b,color:#000
```

---

## 2. Search Data Flow

Shows what happens when a user submits a search query.

```mermaid
sequenceDiagram
    participant U as User (Browser)
    participant F as Next.js Frontend
    participant API as FastAPI Backend
    participant CLIP as CLIP Text Encoder
    participant Cache as LRU Cache (128 entries)
    participant DB as LanceDB

    U->>F: Types query "elephant crossing"
    F->>F: Convert similarity slider to distance threshold<br/>rawSim = CLIP_FLOOR + slider × (CLIP_CEIL - CLIP_FLOOR)<br/>threshold = 1 - rawSim
    F->>API: POST /api/search {query, k, threshold}

    API->>Cache: Check cache for "elephant crossing"
    alt Cache Hit
        Cache-->>API: Return cached 512-dim vector (<1ms)
    else Cache Miss
        API->>CLIP: Encode query text
        CLIP-->>API: 512-dim query vector (~200ms)
        API->>Cache: Store in LRU cache
    end

    par Search 1: Image Vectors
        API->>DB: search(query_vec, column="vector")<br/>cosine metric, limit=k
        DB-->>API: Image results + distances
    and Search 2: Caption Vectors
        API->>DB: search(query_vec, column="caption_vector")<br/>cosine metric, limit=k
        DB-->>API: Caption results + distances
    end

    API->>API: Merge: per photo_id, keep lower distance
    API->>API: Filter: distance ≤ threshold
    API->>API: Rescale: score = (sim - 0.40) / (1.00 - 0.40)
    API-->>F: [{photo_id, photo_image_url, description, score}]

    F->>F: Render result grid with scores and captions
    F-->>U: Display results with "Match %" overlays
```

---

## 3. Ingestion Data Flow (Single Image Upload)

Shows what happens when an admin uploads an image.

```mermaid
sequenceDiagram
    participant A as Admin (Browser)
    participant F as Next.js Admin Panel
    participant API as FastAPI Backend
    participant IS as IngestionService
    participant CLIP as CLIP Image Encoder
    participant BLIP as BLIP Captioner
    participant CLIPT as CLIP Text Encoder
    participant DB as LanceDB
    participant PS as Persistence Service
    participant HF as HF Dataset Repo

    A->>F: Upload image file
    F->>API: POST /api/ingest (multipart/form-data)
    API->>IS: process_file_upload(file)

    IS->>IS: Save image to /data/images/
    IS->>CLIP: embed_image(img)
    CLIP-->>IS: image vector (512-dim)

    IS->>BLIP: generate_caption(img)
    Note over BLIP: Lazy-loads model on<br/>first call (~30s)
    BLIP-->>IS: "a couple of elephants walking down a dirt road"

    IS->>CLIPT: embed_text(caption)
    CLIPT-->>IS: caption vector (512-dim)

    IS->>DB: INSERT ImageRecord<br/>{photo_id, url, description,<br/>vector, caption_vector}

    IS->>PS: trigger background sync
    PS-->>HF: Push updated LanceDB + images<br/>(non-blocking daemon thread)

    API-->>F: {status: "success", description: "a couple of..."}
    F-->>A: Show success + AI-generated caption
```

---

## 4. Ingestion Data Flow (Bulk CSV)

Shows the optimised pipeline for bulk ingestion.

```mermaid
flowchart LR
    subgraph "Phase 1: Download"
        CSV["photos.csv000"] --> PARSE["Parse CSV<br/>(photo_id, url)"]
        PARSE --> POOL["ThreadPoolExecutor<br/>8 concurrent workers"]
        POOL --> IMGS["Downloaded images<br/>(in memory)"]
    end

    subgraph "Phase 2: Embed"
        IMGS --> BATCH["embed_images_batch()<br/>CLIP, batch_size=16"]
        BATCH --> VECS["512-dim vectors"]
    end

    subgraph "Phase 3: Caption (Optional)"
        IMGS --> CAPT{"generate_captions<br/>= True?"}
        CAPT -->|Yes| BLIP["BLIP caption<br/>+ CLIP text embed"]
        CAPT -->|No| ZERO["Zero vector<br/>(512-dim zeros)"]
        BLIP --> CVECS["Caption vectors"]
        ZERO --> CVECS
    end

    subgraph "Phase 4: Store"
        VECS --> INSERT["LanceDB<br/>batch INSERT"]
        CVECS --> INSERT
    end

    subgraph "Phase 5: Index"
        INSERT --> CHECK{"row_count<br/>≥ 256?"}
        CHECK -->|Yes| IVF["Build IVF-PQ Index<br/>partitions=√n, sub_vectors=16"]
        CHECK -->|No| SKIP["Skip indexing<br/>(brute-force OK)"]
    end

    subgraph "Phase 6: Persist"
        INSERT --> SYNC["Background sync<br/>to HF Dataset Repo"]
    end

    style BATCH fill:#4a9eff,color:#fff
    style BLIP fill:#ff6b6b,color:#fff
    style IVF fill:#51cf66,color:#fff
    style POOL fill:#ffd43b,color:#000
```

---

## 5. Dual-Model Architecture (CLIP + BLIP)

Shows how the two AI models work together.

```mermaid
graph TB
    subgraph "At Ingestion Time"
        IMG[Uploaded Image] --> CLIP_IMG["CLIP Image Encoder<br/>(ViT-B/32)"]
        IMG --> BLIP_CAP["BLIP Captioner<br/>(blip-image-captioning-base)"]
        CLIP_IMG --> VEC1["image vector<br/>(512-dim)"]
        BLIP_CAP --> CAP["Caption text<br/>'a couple of elephants<br/>walking down a dirt road'"]
        CAP --> CLIP_TXT["CLIP Text Encoder"]
        CLIP_TXT --> VEC2["caption vector<br/>(512-dim)"]

        VEC1 --> RECORD["ImageRecord in LanceDB"]
        VEC2 --> RECORD
        CAP --> RECORD
    end

    subgraph "At Search Time"
        QUERY["User query:<br/>'elephant crossing'"] --> CLIP_Q["CLIP Text Encoder"]
        CLIP_Q --> QVEC["query vector<br/>(512-dim)"]

        QVEC --> S1["Search 1:<br/>query vs image vectors<br/>(cross-modal)"]
        QVEC --> S2["Search 2:<br/>query vs caption vectors<br/>(same-modal)"]

        S1 --> MERGE["Merge:<br/>best score per image"]
        S2 --> MERGE
        MERGE --> RESULTS["Ranked results<br/>with rescaled scores"]
    end

    style CLIP_IMG fill:#4a9eff,color:#fff
    style CLIP_TXT fill:#4a9eff,color:#fff
    style CLIP_Q fill:#4a9eff,color:#fff
    style BLIP_CAP fill:#ff6b6b,color:#fff
    style MERGE fill:#51cf66,color:#fff
```

---

## 6. Environment Variable Configuration

Shows how the same codebase works locally and deployed.

```mermaid
graph LR
    subgraph "Local Development"
        L_FRONT["localhost:3000<br/>(Next.js dev server)"]
        L_BACK["localhost:8000<br/>(Uvicorn)"]
        L_DATA["./data/<br/>(local filesystem)"]
        L_DB["./lancedb_db/<br/>(local filesystem)"]

        L_FRONT -->|"NEXT_PUBLIC_API_URL<br/>http://localhost:8000/api"| L_BACK
        L_BACK --> L_DATA
        L_BACK --> L_DB
    end

    subgraph "Production Deployment"
        P_FRONT["client-nine-xi-31.vercel.app<br/>(Vercel CDN)"]
        P_BACK["aariz-s-segp-semantic-search.hf.space<br/>(HF Spaces Docker)"]
        P_DATA["/data/images/<br/>(persistent volume)"]
        P_DB["/data/lancedb_db/<br/>(persistent volume)"]
        P_HF["HF Dataset Repo<br/>(Git LFS backup)"]

        P_FRONT -->|"NEXT_PUBLIC_API_URL<br/>https://...hf.space/api"| P_BACK
        P_BACK --> P_DATA
        P_BACK --> P_DB
        P_BACK -->|"HF_TOKEN"| P_HF
    end

    style L_FRONT fill:#4a9eff,color:#fff
    style P_FRONT fill:#4a9eff,color:#fff
    style L_BACK fill:#51cf66,color:#fff
    style P_BACK fill:#51cf66,color:#fff
    style P_HF fill:#ffd43b,color:#000
```
