# System 1: Data Collection & Embedding Generation — Preparation Guide

**For: Karl & Song**
**Prepared by: Aariz**

---

## Your Code Files

You are responsible for understanding these files. Read every line.

### Primary Files (YOUR core responsibility)

| File | Lines | What It Does |
|:---|:---|:---|
| `server/app/services/search_service.py` | 147 | CLIP model loading, text/image embedding, LRU cache, batch embedding |
| `server/app/services/caption_service.py` | 41 | BLIP model loading, image captioning |
| `server/app/services/ingestion_service.py` | 184 | The pipeline that ties everything together — upload processing, schema definition, bulk ingestion |

### Supporting Files (you should be familiar with these)

| File | Lines | Why It Matters to You |
|:---|:---|:---|
| `server/app/main.py` | 58 | Server startup — loads CLIP at boot, restores data from HF repo |
| `server/app/routers/upload.py` | 66 | The API endpoints that trigger your ingestion pipeline |
| `client/app/admin/page.tsx` | 419 | The admin panel UI — where uploads happen and captions display |

---

## Relevant Code Sections (In Detail)

### A. CLIP Model Initialisation (`search_service.py`, lines 19-33)

```python
self.model_name = os.getenv("EMBED_MODEL", "openai/clip-vit-base-patch32")
self.device = "cuda" if torch.cuda.is_available() else "cpu"
self._text_cache = OrderedDict()  # LRU cache for text embeddings
self._text_cache_max = 128
self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
self.processor = CLIPProcessor.from_pretrained(self.model_name)
self.model.eval()
```

- The model is `openai/clip-vit-base-patch32` — a Vision Transformer (ViT) with 32x32 patch size
- Runs on GPU if available, CPU otherwise (our deployment is CPU-only)
- `model.eval()` disables dropout — we only do inference, never training
- The LRU cache stores up to 128 text embeddings to avoid re-computing repeated queries

### B. Text Embedding with LRU Cache (`search_service.py`, lines 50-70)

```python
@torch.no_grad()
def embed_text(self, text: str) -> np.ndarray:
    cache_key = text.strip().lower()
    if cache_key in self._text_cache:
        self._text_cache.move_to_end(cache_key)
        return self._text_cache[cache_key]

    inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
    text_features = self.model.get_text_features(**inputs)
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    result = text_features.cpu().numpy().astype("float32")[0]

    self._text_cache[cache_key] = result
    if len(self._text_cache) > self._text_cache_max:
        self._text_cache.popitem(last=False)
    return result
```

- `@torch.no_grad()` disables gradient tracking — saves ~50% memory, ~20% faster
- Cache key is normalised: "Sunset", "sunset", " SUNSET " all map to the same entry
- `move_to_end()` marks a cache entry as recently used (LRU policy)
- `popitem(last=False)` evicts the least recently used entry when cache is full
- L2 normalisation (`/ norm()`) ensures all vectors have unit length — required for cosine similarity
- `pooler_output` handling is a compatibility fix for different `transformers` library versions

### C. Image Embedding (`search_service.py`, lines 72-81)

```python
@torch.no_grad()
def embed_image(self, image: Image.Image) -> np.ndarray:
    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    image_features = self.model.get_image_features(**inputs)
    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype("float32")[0]
```

- Same model, different method: `get_image_features()` vs `get_text_features()`
- CLIP's architecture has two encoders: a Vision Transformer for images, a text Transformer for text
- Both produce 512-dimensional vectors in the SAME space — that is what makes cross-modal search possible

### D. Batch Image Embedding (`search_service.py`, lines 84-97)

```python
@torch.no_grad()
def embed_images_batch(self, images: list[Image.Image], batch_size: int = 16) -> list[np.ndarray]:
    all_vectors = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        vectors = image_features.cpu().numpy().astype("float32")
        all_vectors.extend([vectors[j] for j in range(len(batch))])
    return all_vectors
```

- Processes 16 images at once through CLIP instead of one at a time
- The Transformer architecture benefits from batching — per-image cost drops significantly
- Used during bulk CSV ingestion (1000+ images at a time)

### E. BLIP Caption Service (`caption_service.py`, entire file)

```python
class CaptionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CaptionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _ensure_loaded(self):
        if self._initialized:
            return
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._initialized = True

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        self._ensure_loaded()
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
```

- Singleton pattern (`__new__` override) — only one ~990MB model in memory
- Lazy loading (`_ensure_loaded`) — the model does NOT load at server startup, only when the first caption is requested. This cuts cold start time from ~60s to ~30s for search-only sessions.
- The `from transformers import ...` is INSIDE `_ensure_loaded()`, not at the top of the file — even the import is deferred because `transformers` is slow to import
- RGB conversion guard — BLIP crashes on RGBA/palette-mode images without this
- `model.generate()` is autoregressive — it produces tokens one at a time until it generates a stop token

### F. ImageRecord Schema (`ingestion_service.py`, lines 18-23)

```python
class ImageRecord(LanceModel):
    photo_id: str
    photo_image_url: str
    description: str = ""
    vector: Vector(512)
    caption_vector: Vector(512)
```

- This is the database schema. Every image has these 5 fields.
- `vector` = CLIP image embedding (512 floats)
- `caption_vector` = CLIP text embedding of the BLIP caption (512 floats)
- `description` = the BLIP caption string (e.g., "a couple of elephants walking down a dirt road")

### G. Single Image Upload Pipeline (`ingestion_service.py`, lines 42-68)

```python
def process_upload(self, file_contents: bytes, filename: str):
    file_path = self.data_dir / filename
    with open(file_path, "wb") as f:
        f.write(file_contents)

    img = Image.open(file_path).convert("RGB")

    vector = search_service.embed_image(img)                 # Step 1: CLIP image → vector
    description = caption_service.generate_caption(img)      # Step 2: BLIP image → caption
    caption_vector = search_service.embed_text(description)  # Step 3: CLIP text → vector

    photo_url = f"/images/{filename}"
    record = ImageRecord(
        photo_id=file_path.stem,
        photo_image_url=photo_url,
        description=description,
        vector=vector,
        caption_vector=caption_vector,
    )
    self._insert_records([record])
    sync_to_repo()
    return {"id": file_path.stem, "url": photo_url, "description": description, "status": "indexed"}
```

- The 3-step pipeline: CLIP(image) → BLIP(image) → CLIP(caption text)
- `sync_to_repo()` pushes to the HF Dataset repository after every upload (persistence)

### H. Bulk CSV Ingestion (`ingestion_service.py`, lines 94-149)

```python
def process_bulk_csv(self, csv_path, limit=100, generate_captions=False, max_workers=8, batch_size=16):
    # Phase 1: Concurrent downloads (8 threads)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_image, pid, url): i for i, (pid, url) in enumerate(rows)}
        for future in as_completed(futures):
            photo_id, url, img = future.result()
            downloaded.append((photo_id, url, img))

    # Phase 2: Batch embed via CLIP (16 images per forward pass)
    images = [img for _, _, img in downloaded]
    vectors = search_service.embed_images_batch(images, batch_size=batch_size)

    # Phase 3: Optional BLIP captioning
    zero_vec = np.zeros(512, dtype="float32")
    for i, (photo_id, url, img) in enumerate(downloaded):
        description = ""
        cap_vec = zero_vec
        if generate_captions:
            description = caption_service.generate_caption(img)
            cap_vec = search_service.embed_text(description)
        records.append(ImageRecord(...))

    # Phase 4: Insert all at once
    self._insert_records(records)

    # Phase 5: Maybe build IVF-PQ index
    self._maybe_create_index()

    # Phase 6: Persist
    sync_to_repo()
```

- Captioning is OFF by default for bulk (`generate_captions=False`) — it takes ~200ms per image
- When skipped, `caption_vector` is a zero vector (512 zeros) — hybrid search will ignore it
- ThreadPoolExecutor downloads 8 images simultaneously (network-bound, not CPU-bound)

### I. Upload Endpoints (`upload.py`, lines 1-66)

```python
@router.post("/upload")
async def upload_image(file: UploadFile):
    result = ingestion_service.process_upload(await file.read(), file.filename)
    # Returns: {id, url, description, status}

@router.post("/ingest/url")
async def ingest_url(req: UrlIngestRequest):
    result = ingestion_service.process_url_upload(req.url, req.photo_id)

@router.post("/ingest/bulk")
async def bulk_ingest(req: BulkIngestRequest):
    result = ingestion_service.process_bulk_csv(req.csv_path, req.limit, req.generate_captions)

@router.delete("/clear-db")
async def clear_database():
    # Wipes the LanceDB table and syncs the empty state
```

---

## Topic List: Everything You Need to Understand

### CLIP (Contrastive Language-Image Pre-training)
1. What CLIP is and who made it (OpenAI, 2021)
2. How CLIP's dual-encoder architecture works (Vision Transformer for images, text Transformer for text)
3. What "shared embedding space" means and why it enables cross-modal search
4. What a 512-dimensional vector is and what each dimension represents (learned features, not human-interpretable)
5. What L2 normalisation is and why we do it (unit vectors → cosine similarity = dot product)
6. What `@torch.no_grad()` does and why we use it (inference only, no training)
7. The difference between `get_text_features()` and `get_image_features()`
8. What the LRU cache does and why it matters for performance
9. What batch embedding is and why it is faster than one-at-a-time
10. The `pooler_output` compatibility issue with newer `transformers` versions

### BLIP (Bootstrapping Language-Image Pre-training)
11. What BLIP is and who made it (Salesforce, 2022)
12. How BLIP differs from CLIP (generates text vs generates vectors)
13. Why we need BLIP in addition to CLIP (human-readable captions, hybrid search, accessibility)
14. The specific model variant we use (`blip-image-captioning-base`, ~990MB)
15. The singleton pattern and why only one instance exists
16. Lazy loading — why the model loads on first use, not at startup
17. Why the import statement is inside `_ensure_loaded()` not at the top of the file
18. The RGB conversion guard and why it is needed
19. Known BLIP quality issues (repetition loops, shallow descriptions)
20. Why we chose BLIP over Gemini (CLIP compatibility, speed, no external dependency)

### The Ingestion Pipeline
21. The 3-step pipeline: CLIP(image) → BLIP(image) → CLIP(caption)
22. Why two vectors are stored per image (hybrid search — visual + textual matching)
23. The ImageRecord schema (5 fields, 2 vector columns)
24. How single file upload works (synchronous, always captioned)
25. How URL ingestion works (downloads image first, then same pipeline)
26. How bulk CSV ingestion works (6 phases: download → embed → caption → insert → index → persist)
27. Why captioning is optional for bulk (speed — ~200ms per image adds up)
28. What happens when captions are skipped (zero vector → caption search returns nothing for that image)
29. Concurrent downloading with ThreadPoolExecutor (8 workers, network-bound)
30. `sync_to_repo()` — background persistence to HF Dataset repository

---

## 10 Questions You Must Be Able to Answer

1. **What AI model generates the image embeddings, and what is the output?**
2. **What AI model generates the image captions, and give an example of its output?**
3. **Walk me through what happens step-by-step when a user uploads a single image.**
4. **Why do we store two vectors per image instead of one?**
5. **What is the LRU cache and why does it improve performance?**
6. **Why is BLIP lazy-loaded instead of loading at server startup?**
7. **How does bulk ingestion differ from single upload? Name three specific optimisations.**
8. **Why did we choose BLIP over Google Gemini for captioning?**
9. **What does `@torch.no_grad()` do and why is it on every embedding method?**
10. **What is L2 normalisation and why do we apply it to every vector?**
