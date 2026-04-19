# Open Day Cheat Sheets

Hand the relevant section to each pair. They need to read the code, understand the statement, and be able to answer the questions.

---

## System 1: Data Collection & Embedding Generation (Karl & Song)

### Your Code

**File 1: `server/app/services/search_service.py` (lines 19-81)**

This is where CLIP lives. Read these three methods:

```python
# Lines 19-33 — Model loading
self.model_name = "openai/clip-vit-base-patch32"
self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
self.processor = CLIPProcessor.from_pretrained(self.model_name)
```

```python
# Lines 50-70 — Text embedding (with LRU cache)
def embed_text(self, text: str) -> np.ndarray:
    cache_key = text.strip().lower()
    if cache_key in self._text_cache:
        return self._text_cache[cache_key]  # Cache hit — instant return

    inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    text_features = self.model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # L2 normalise
    return text_features.cpu().numpy().astype("float32")[0]  # 512-dim vector
```

```python
# Lines 72-81 — Image embedding
def embed_image(self, image: Image.Image) -> np.ndarray:
    inputs = self.processor(images=image, return_tensors="pt")
    image_features = self.model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype("float32")[0]  # 512-dim vector
```

**File 2: `server/app/services/caption_service.py` (entire file)**

This is where BLIP lives. It generates natural language captions from images.

```python
class CaptionService:
    # Singleton — only one instance of the ~990MB model in memory
    # Lazy-loaded — doesn't load until first caption is requested

    def _ensure_loaded(self):
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)

    def generate_caption(self, image) -> str:
        self._ensure_loaded()
        inputs = self.processor(images=image, return_tensors="pt")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
        # Returns e.g. "a couple of elephants walking down a dirt road"
```

**File 3: `server/app/services/ingestion_service.py` (lines 42-68)**

This is where your models get called during image upload:

```python
def process_upload(self, file_contents, filename):
    img = Image.open(file_path).convert("RGB")

    vector = search_service.embed_image(img)                 # Step 1: CLIP image -> 512-dim vector
    description = caption_service.generate_caption(img)      # Step 2: BLIP image -> caption text
    caption_vector = search_service.embed_text(description)  # Step 3: CLIP text -> 512-dim vector

    record = ImageRecord(
        photo_id=file_path.stem,
        photo_image_url=photo_url,
        description=description,      # "a couple of elephants walking..."
        vector=vector,                 # CLIP image embedding
        caption_vector=caption_vector, # CLIP text embedding of BLIP caption
    )
```

### What You Need to Understand

Your subsystem handles converting raw images into searchable data. Two AI models work together:

- **CLIP** (by OpenAI) converts both images and text into 512-dimensional vectors in a shared space. This is what makes text-to-image search possible — the text query and the image end up in the same mathematical space, so we can measure how close they are.

- **BLIP** (by Salesforce) looks at an image and generates a natural language caption — e.g., "a golden retriever running on the beach." This caption is then also embedded by CLIP to create a second search vector per image.

Every uploaded image goes through a 3-step pipeline: CLIP embeds the image, BLIP captions it, then CLIP embeds the caption text. Two vectors are stored per image.

### Questions You Should Be Able to Answer

1. **"What model do you use for image search?"**
   - "We use CLIP — specifically the ViT-B/32 variant by OpenAI. It converts both images and text into 512-dimensional vectors in a shared embedding space."

2. **"What is BLIP and why do you need it?"**
   - "BLIP is an image captioning model by Salesforce. It generates natural language descriptions of images. We use it because CLIP alone can only produce vectors — BLIP gives us human-readable captions that we display on hover and also use to improve search accuracy."

3. **"How does an image get indexed?"**
   - "Three steps: CLIP generates a vector from the image, BLIP generates a text caption, then CLIP generates a second vector from that caption. Both vectors and the caption are stored in the database."

4. **"Why two vectors per image?"**
   - "One is from the image (for visual matching), one is from the caption text (for text matching). When you search, we compare your query against both and keep the better match. This is called hybrid search."

5. **"Why didn't you use Gemini instead of BLIP?"**
   - "We tested it. Gemini produces much better captions, but CLIP's text encoder was trained on short, simple captions — the kind BLIP produces. Gemini's long, detailed captions actually diluted the embeddings. BLIP and CLIP are from the same era and work better together. Our benchmarks confirmed BLIP gives higher search accuracy."

6. **"What is the embedding dimension and why 512?"**
   - "512 is the output dimension of CLIP ViT-B/32. It's a fixed property of the model architecture — the Vision Transformer produces 512-dimensional feature vectors."

---

## System 2: Indexing & Retrieval (Bessie & Zheng)

### Your Code

**File 1: `server/app/services/ingestion_service.py` (lines 18-23, 151-180)**

This is the database schema and index creation:

```python
# Lines 18-23 — What gets stored per image
class ImageRecord(LanceModel):
    photo_id: str                  # Unique ID
    photo_image_url: str           # Path to the image file
    description: str = ""          # BLIP-generated caption
    vector: Vector(512)            # CLIP image embedding
    caption_vector: Vector(512)    # CLIP text embedding of the caption
```

```python
# Lines 160-180 — IVF-PQ index creation (after bulk ingestion)
def _maybe_create_index(self, min_rows=256):
    row_count = table.count_rows()
    if row_count < min_rows:
        return  # Brute-force is fine for small tables

    num_partitions = max(2, int(row_count ** 0.5))  # sqrt(n) partitions
    table.create_index(
        metric="cosine",
        num_partitions=num_partitions,
        num_sub_vectors=16,
        replace=True,
    )
```

**File 2: `server/app/services/search_service.py` (lines 99-144)**

This is the hybrid search — the core retrieval logic:

```python
def search(self, query, k=20, threshold=0.9):
    query_vec = self.embed_text(query)

    # Search 1: query vs IMAGE vectors (visual similarity)
    image_results = (
        self.table.search(query_vec, vector_column_name="vector")
        .metric("cosine").limit(k).to_pandas()
    )

    # Search 2: query vs CAPTION vectors (text similarity)
    caption_results = (
        self.table.search(query_vec, vector_column_name="caption_vector")
        .metric("cosine").limit(k).to_pandas()
    )

    # Merge: for each image, keep the better (lower distance) score
    best = {}
    for df in [image_results, caption_results]:
        for _, row in df.iterrows():
            pid = row["photo_id"]
            if pid not in best or row["_distance"] < best[pid]["_distance"]:
                best[pid] = row.to_dict()

    results = sorted(best.values(), key=lambda r: r["_distance"])
    results = [r for r in results if r["_distance"] <= threshold]
    return results[:k]
```

**File 3: `server/app/routers/search.py` (lines 11-20)**

This is how raw distances become user-friendly percentages:

```python
CLIP_SIM_FLOOR = 0.40  # Below this = 0%
CLIP_SIM_CEIL = 1.00   # Exact match = 100%

def rescale_clip_score(raw_cosine_distance: float) -> float:
    raw_sim = 1.0 - raw_cosine_distance      # Convert distance to similarity
    scaled = (raw_sim - 0.40) / (1.00 - 0.40) # Normalise to 0-1
    return max(0.0, min(1.0, scaled))          # Clamp
```

### What You Need to Understand

Your subsystem handles storing vectors and finding the closest matches. The key concepts:

- **LanceDB** is our vector database. It is embedded (runs inside our Python process, no separate server) and stores both the vector data and metadata together. Think of it like SQLite but for vectors.

- **Cosine distance** measures how different two vectors are. 0.0 = identical, 2.0 = opposite. We use this to rank search results — lower distance = better match.

- **Hybrid search** runs two searches per query: one against image vectors, one against caption vectors. For each image, we keep whichever search gave the better (lower distance) result. This is why we store two vectors per image.

- **IVF-PQ index** is created automatically when the database has 256+ images. It divides the vector space into clusters so we don't have to scan every single vector on every search. Without it, search is O(n) — with it, it's much faster.

- **Score rescaling** converts raw cosine distances (which are meaningless to users) into 0-100% scores. CLIP's raw output range doesn't map intuitively to percentages, so we normalise it.

### Questions You Should Be Able to Answer

1. **"What database do you use?"**
   - "LanceDB — it's an open-source, embedded vector database. It runs inside our FastAPI server process, so there's no separate database server to manage. It stores both the vectors and the metadata together."

2. **"How does the search work?"**
   - "When you type a query, CLIP converts it to a 512-dimensional vector. We then search LanceDB for the closest vectors using cosine distance. We actually run two searches — one against the image vectors and one against the caption vectors — and merge the results, keeping the best match for each image."

3. **"What is cosine distance?"**
   - "It measures the angle between two vectors. 0 means identical direction, 2 means opposite. For search, lower distance means a better match. We use it instead of Euclidean distance because it works better for normalised embeddings."

4. **"Why do the scores show percentages instead of raw distances?"**
   - "Because raw cosine distances are not intuitive. A 'good' match might be 0.3 distance and a 'bad' match 0.7 — that doesn't mean much to a user. We rescale the range so 0% means unrelated and 100% means a perfect match."

5. **"What is the IVF-PQ index?"**
   - "It stands for Inverted File Index with Product Quantisation. It divides the vector space into clusters so that during search, we only check the nearest clusters instead of every single vector. It makes search much faster for large datasets — we create it automatically when the database has more than 256 images."

6. **"How does data persist if the server restarts?"**
   - "We sync the LanceDB files and uploaded images to a Hugging Face Dataset repository in the background. When the server starts up, it checks that repo and downloads any existing data before accepting requests."

---

## Shared Knowledge (Everyone Should Know This)

### The One-Liner
"Our project is a semantic image search engine — you type a description in natural language, and it finds the most visually and semantically similar images using AI."

### The Tech Stack
- **Frontend:** Next.js + React (hosted on Vercel)
- **Backend:** FastAPI / Python (hosted on Hugging Face Spaces)
- **AI Models:** CLIP (search embeddings) + BLIP (image captioning)
- **Database:** LanceDB (vector database)
- **Deployment:** Docker container on HF Spaces, with HF Dataset repo for persistence

### The Demo Script
1. Go to the search page
2. Type "elephant crossing" — show the results with match percentages
3. Hover over a result to show the BLIP caption
4. Go to the admin panel
5. Upload a new image — show the caption that gets generated
6. Search for that image — show it appears in results
7. Show the "See All Indexed Images" tab

### If Someone Asks Something You Don't Know
Say: "That's handled by [other subsystem] — let me get [name] to explain that part." Do not guess. It is always better to redirect than to say something wrong.
