# System 2: Indexing & Retrieval — Preparation Guide

**For: Bessie & Zheng**
**Prepared by: Aariz**

---

## Your Code Files

You are responsible for understanding these files. Read every line.

### Primary Files (YOUR core responsibility)

| File | Lines | What It Does |
|:---|:---|:---|
| `server/app/services/search_service.py` | 147 | The hybrid search algorithm, LanceDB table management, vector comparison |
| `server/app/routers/search.py` | 80 | Search API endpoint, score rescaling, the `/images/all` listing endpoint |
| `server/app/services/ingestion_service.py` | 184 | Database schema (`ImageRecord`), record insertion (`_insert_records`), IVF-PQ index creation (`_maybe_create_index`) |

### Supporting Files (you should be familiar with these)

| File | Lines | Why It Matters to You |
|:---|:---|:---|
| `server/app/main.py` | 58 | Server startup — restores LanceDB data from HF repo, mounts static files |
| `server/app/routers/upload.py` | 66 | Upload endpoints that trigger database writes |
| `client/app/page.tsx` | 310 | Frontend search UI — sends queries to your search endpoint, displays your results |
| `client/lib/api.ts` | 97 | API client — shows how the frontend calls your endpoints |
| `client/app/admin/page.tsx` | 419 | Admin panel — the "All Images" gallery tab calls your `/images/all` endpoint |

---

## Relevant Code Sections (In Detail)

### A. Database Schema (`ingestion_service.py`, lines 18-23)

```python
class ImageRecord(LanceModel):
    photo_id: str                  # Unique identifier (filename stem or CSV ID)
    photo_image_url: str           # Relative path: "/images/filename.jpg"
    description: str = ""          # BLIP-generated caption text
    vector: Vector(512)            # CLIP image embedding (512 floats)
    caption_vector: Vector(512)    # CLIP text embedding of the BLIP caption (512 floats)
```

- This is a Pydantic model that doubles as a LanceDB schema (via `LanceModel`)
- `Vector(512)` tells LanceDB to store a fixed-size array of 512 float32 values
- Two vector columns enable dual-pathway hybrid search
- `description` defaults to empty string — bulk-ingested images without captions have `""` here and a zero vector for `caption_vector`

### B. Record Insertion (`ingestion_service.py`, lines 151-158)

```python
def _insert_records(self, records: List[ImageRecord]):
    db = lancedb.connect(search_service.db_uri)
    try:
        table = db.open_table("images")
        table.add(records)
    except:
        db.create_table("images", schema=ImageRecord, data=records)
    search_service.refresh_table()
```

- First attempts to open existing "images" table and add records
- If the table doesn't exist yet, creates it with the ImageRecord schema
- `refresh_table()` updates the SearchService's reference to the table so new data is immediately searchable
- `lancedb.connect()` takes a URI — local path for development, `/data/lancedb_db` in production

### C. IVF-PQ Index Creation (`ingestion_service.py`, lines 160-180)

```python
def _maybe_create_index(self, min_rows: int = 256):
    try:
        table = search_service.table
        if table is None:
            return
        row_count = table.count_rows()
        if row_count < min_rows:
            print(f"Skipping index creation: {row_count} rows < {min_rows} minimum")
            return
        num_partitions = max(2, int(row_count ** 0.5))
        num_sub_vectors = 16
        print(f"Building IVF-PQ index: {row_count} rows, {num_partitions} partitions, {num_sub_vectors} sub-vectors...")
        table.create_index(
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            replace=True,
        )
        print("IVF-PQ index built successfully.")
    except Exception as e:
        print(f"Index creation skipped or failed: {e}")
```

- Called after bulk CSV ingestion only (not after single uploads — too few records to justify)
- `min_rows=256`: below this threshold, brute-force scanning is faster than index overhead
- `num_partitions = sqrt(row_count)`: a standard heuristic — balances granularity vs overhead
- `num_sub_vectors = 16`: each 512-dim vector is split into 16 groups of 32 dimensions for compression
- `metric="cosine"`: must match the metric used in search queries
- `replace=True`: rebuilds the index if one already exists (important after adding more data)

### D. LanceDB Table Management (`search_service.py`, lines 35-48)

```python
def refresh_table(self):
    try:
        db = lancedb.connect(self.db_uri)
        self.table = db.open_table("images")
        print(f"Table loaded: {self.table.count_rows()} rows")
    except Exception:
        self.table = None
        print("No existing table found.")
```

- Called at startup and after every record insertion
- `self.table` is the live reference used by the `search()` method
- If no table exists yet (fresh server), `self.table = None` and search returns empty results

### E. Hybrid Search Algorithm (`search_service.py`, lines 99-144)

```python
def search(self, query: str, k: int = 20, threshold: float = 0.9):
    if self.table is None:
        self.refresh_table()
        if self.table is None:
            return []

    query_vec = self.embed_text(query)
    select_cols = ["photo_id", "photo_image_url", "description", "_distance"]

    # Search 1: query text vector vs image vectors (cross-modal)
    image_results = (
        self.table.search(query_vec, vector_column_name="vector")
        .metric("cosine")
        .limit(k)
        .select(select_cols)
        .to_pandas()
    )

    # Search 2: query text vector vs caption vectors (same-modal)
    caption_results = (
        self.table.search(query_vec, vector_column_name="caption_vector")
        .metric("cosine")
        .limit(k)
        .select(select_cols)
        .to_pandas()
    )

    # Merge: for each image, keep the better (lower distance) match
    best = {}
    for df in [image_results, caption_results]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            pid = row["photo_id"]
            dist = row.get("_distance", 1.0)
            if pid not in best or dist < best[pid]["_distance"]:
                best[pid] = row.to_dict()

    if not best:
        return []

    results = sorted(best.values(), key=lambda r: r["_distance"])
    results = [r for r in results if r["_distance"] <= threshold]
    return results[:k]
```

- **Search 1 (cross-modal):** Text query vector compared against image embedding vectors. This is text-to-image matching. CLIP was trained to align text and images, but they are different modalities — max similarity is typically ~0.35.
- **Search 2 (same-modal):** Text query vector compared against caption text vectors. This is text-to-text matching. Because both are text embeddings from CLIP's text encoder, similarity can reach 1.0 for exact matches.
- **Merge strategy:** For each `photo_id`, keep whichever search gave the lower `_distance` (= better match). This means an image can match via either pathway.
- **Threshold filter:** Removes results with distance above the threshold (default 0.9 = very permissive).
- **`_distance`:** LanceDB automatically adds this column — it is the cosine distance between the query vector and the stored vector.

### F. Score Rescaling (`search.py`, lines 11-20)

```python
CLIP_SIM_FLOOR = 0.40  # Below this = 0% (unrelated)
CLIP_SIM_CEIL = 1.00   # Exact caption match = 100%

def rescale_clip_score(raw_cosine_distance: float) -> float:
    raw_sim = 1.0 - raw_cosine_distance      # Convert distance → similarity
    scaled = (raw_sim - CLIP_SIM_FLOOR) / (CLIP_SIM_CEIL - CLIP_SIM_FLOOR)
    return max(0.0, min(1.0, scaled))          # Clamp to [0, 1]
```

**Why rescaling is necessary:**
- Cosine distance ranges from 0.0 (identical) to 2.0 (opposite)
- CLIP's actual output for cross-modal comparisons sits in a narrow band (~0.10-0.35 similarity)
- After hybrid search, caption matches can reach 1.0 similarity
- Without rescaling, a "good" match shows as 22% — users think the system is broken
- The rescaling maps the actual useful range (0.40-1.00 similarity) to 0-100%

**Worked examples:**
| Raw Distance | Raw Similarity | Rescaled Score | Meaning |
|:---|:---|:---|:---|
| 0.00 | 1.00 | 100% | Perfect caption match |
| 0.20 | 0.80 | 67% | Strong match |
| 0.40 | 0.60 | 33% | Moderate match |
| 0.60 | 0.40 | 0% | Below floor — filtered out |

### G. Search API Endpoint (`search.py`, lines 34-56)

```python
@router.post("/search", response_model=List[SearchResult])
async def search_images(req: SearchRequest):
    results = search_service.search(req.query, req.k, req.threshold)
    response = []
    for r in results:
        dist = r.get("_distance", 1.0)
        score = rescale_clip_score(dist)
        url = r["photo_image_url"]
        if url.startswith("/images/"):
            url = f"{BACKEND_URL}{url}"
        response.append({
            "photo_id": r["photo_id"],
            "photo_image_url": url,
            "description": r.get("description", ""),
            "score": float(score)
        })
    return response
```

- Takes `SearchRequest` (query string, k limit, distance threshold)
- Calls the hybrid search, rescales each result's distance to a percentage
- Converts relative URLs (`/images/...`) to absolute URLs using `BACKEND_URL`
- Returns a list of `SearchResult` objects with `photo_id`, `photo_image_url`, `description`, `score`

### H. List All Indexed Images (`search.py`, lines 57-79)

```python
@router.get("/images/all")
async def list_all_images():
    table = search_service.table
    if table is None:
        return []
    df = table.to_pandas()
    results = []
    for _, row in df.iterrows():
        url = row.get("photo_image_url", "")
        if isinstance(url, str) and url.startswith("/images/"):
            url = f"{BACKEND_URL}{url}"
        results.append({
            "photo_id": row.get("photo_id", ""),
            "photo_image_url": url,
            "description": row.get("description", ""),
        })
    return results
```

- This endpoint powers the "All Images" gallery tab in the admin panel
- Reads the entire LanceDB table into a pandas DataFrame
- Returns every record with its ID, URL, and BLIP caption
- No search involved — just a full table dump

### I. Frontend Search Parameters (`page.tsx`, lines 24-31)

```typescript
const CLIP_FLOOR = 0.40;
const CLIP_CEIL = 1.00;

// Convert user's similarity slider (0-100%) to backend distance threshold
const rawSim = CLIP_FLOOR + minSimilarity * (CLIP_CEIL - CLIP_FLOOR);
const distanceThreshold = 1 - rawSim;
```

- The frontend mirrors the backend's rescaling constants
- When user sets similarity slider to 50%, the backend receives a distance threshold
- This ensures the slider and the displayed scores are consistent

### J. Frontend API Client (`api.ts`, key functions)

```typescript
export async function searchImages(query: string, k: number, threshold: number) {
    const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, k, threshold }),
    });
    return res.json();
}

export async function listAllImages() {
    const res = await fetch(`${API_BASE}/images/all`);
    return res.json();
}
```

- `searchImages()` sends the query, max results (k), and distance threshold to your search endpoint
- `listAllImages()` fetches every indexed image for the admin gallery tab

### K. Persistence: Server Startup (`main.py`, lines 10-20)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    restore_from_repo()  # Download LanceDB + images from HF Dataset repo
    search_service.refresh_table()  # Load the table into memory
    yield
```

- On startup, before accepting any requests, the server downloads persisted data from the HF Dataset repository
- This is how the database survives server restarts on HF Spaces (which has ephemeral storage)
- After restore, `refresh_table()` loads the LanceDB table so search works immediately

---

## Topic List: Everything You Need to Understand

### LanceDB (The Vector Database)
1. What LanceDB is (embedded, open-source vector database — like SQLite for vectors)
2. How it differs from traditional databases (stores high-dimensional vectors, supports similarity search)
3. How it differs from other vector databases (Pinecone is cloud-hosted, FAISS is a library not a DB — LanceDB is embedded with metadata support)
4. The `ImageRecord` schema — what each of the 5 fields stores
5. How records are inserted (`_insert_records` — try open, fall back to create)
6. How the table is loaded at startup (`refresh_table`)
7. What `lancedb.connect(uri)` does (opens or creates a database at the given path)
8. How data persists across server restarts (HF Dataset repo sync)

### Cosine Distance & Similarity
9. What cosine distance is (measures the angle between two vectors — 0 = identical, 2 = opposite)
10. The relationship between distance and similarity: `similarity = 1 - distance`
11. Why we use cosine distance instead of Euclidean distance (works better for normalised embeddings, invariant to vector magnitude)
12. Why all vectors are L2-normalised before storage (ensures cosine similarity = dot product, more efficient)
13. Why CLIP's cross-modal similarity is low (~0.20-0.35) even for good matches
14. Why same-modal similarity (text vs text caption) can reach 1.0

### The Hybrid Search Algorithm
15. What "hybrid search" means in this system (dual-vector search, not keyword + vector)
16. Search 1: query vs image vectors — what it captures (visual similarity)
17. Search 2: query vs caption vectors — what it captures (semantic/textual similarity)
18. The merge strategy: per-`photo_id`, keep the lower distance
19. Why the merge uses "best of two" instead of averaging (averaging would weaken strong single-pathway matches)
20. The threshold filter — what it does and what the default 0.9 means
21. How results are sorted (ascending by distance = best match first)
22. What happens for uncaptioned images (zero vector → caption search returns high distance → image search pathway still works)

### Score Rescaling
23. Why raw cosine similarity cannot be shown to users (22% for a correct match looks broken)
24. The CLIP_SIM_FLOOR (0.40) and CLIP_SIM_CEIL (1.00) — what they represent
25. The rescaling formula: `(similarity - floor) / (ceil - floor)`, clamped to [0, 1]
26. How the frontend slider maps back to a distance threshold
27. Why the constants changed from 0.10/0.35 to 0.40/1.00 (hybrid search shifted the output range)

### IVF-PQ Indexing
28. What IVF means (Inverted File Index — divides vector space into partitions/clusters)
29. What PQ means (Product Quantisation — compresses vectors by splitting into sub-vectors)
30. How IVF-PQ makes search faster (only scan nearest partitions instead of all rows)
31. Why the index is only created at 256+ rows (overhead not worth it for small tables)
32. Why `num_partitions = sqrt(row_count)` (standard heuristic balancing granularity vs overhead)
33. Why `num_sub_vectors = 16` (512 dims / 16 = 32 dims per sub-vector)
34. The trade-off: IVF-PQ is approximate — it may miss some results that brute-force would find

### The `/images/all` Endpoint
35. What the "All Images" gallery tab shows (every indexed image with its BLIP caption)
36. How it works technically (full table dump to pandas DataFrame)
37. Why this is useful for admins (verify captions, check database contents, spot-check BLIP output)

---

## 10 Questions You Must Be Able to Answer

1. **What database does the system use and why was it chosen over alternatives?**
2. **What is cosine distance and how does it relate to similarity?**
3. **Walk me through what happens when a user searches for "elephant crossing" — from the moment they hit Enter to the moment results appear.**
4. **What is hybrid search and why does it produce better results than single-vector search?**
5. **Why do we rescale the scores, and what would happen if we showed raw similarity values?**
6. **What is the IVF-PQ index, and why is it only created when we have 256+ images?**
7. **What is the difference between Search 1 (image vectors) and Search 2 (caption vectors)?**
8. **How does the system handle images that were bulk-ingested without captions?**
9. **How does the LanceDB data survive server restarts on Hugging Face Spaces?**
10. **What does the "All Images" tab in the admin panel show, and what endpoint does it call?**
