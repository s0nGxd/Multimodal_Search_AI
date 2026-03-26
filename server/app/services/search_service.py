import os
import hashlib
from collections import OrderedDict
import lancedb
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class SearchService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Initializing SearchService...")
        self.db_uri = os.getenv("LANCEDB_URI", "./lancedb_db")
        self.model_name = os.getenv("EMBED_MODEL", "openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Text embedding cache (LRU, max 128 entries)
        self._text_cache = OrderedDict()
        self._text_cache_max = 128

        # Load Model
        print(f"Loading CLIP model: {self.model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()

        # Connect to DB
        self.db = lancedb.connect(self.db_uri)
        try:
            self.table = self.db.open_table("images")
        except:
            print("Table 'images' not found. It will need to be created via ingestion.")
            self.table = None

    def refresh_table(self):
        """Re-opens the table in case it was created/updated."""
        try:
            self.table = self.db.open_table("images")
        except:
            self.table = None

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

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)

        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype("float32")[0]

    @torch.no_grad()
    def embed_images_batch(self, images: list[Image.Image], batch_size: int = 16) -> list[np.ndarray]:
        all_vectors = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            image_features = self.model.get_image_features(**inputs)

            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            vectors = image_features.cpu().numpy().astype("float32")
            all_vectors.extend([vectors[j] for j in range(len(batch))])
        return all_vectors

    def search(self, query: str, k: int = 20, threshold: float = 0.9):
        if self.table is None:
            self.refresh_table()
            if self.table is None:
                return []

        query_vec = self.embed_text(query)
        select_cols = ["photo_id", "photo_image_url", "description", "_distance"]

        # Search 1: query vs image vectors (visual similarity)
        image_results = (
            self.table.search(query_vec, vector_column_name="vector")
            .metric("cosine")
            .limit(k)
            .select(select_cols)
            .to_pandas()
        )

        # Search 2: query vs caption vectors (text similarity)
        caption_results = (
            self.table.search(query_vec, vector_column_name="caption_vector")
            .metric("cosine")
            .limit(k)
            .select(select_cols)
            .to_pandas()
        )

        # Merge: for each image, take the lower distance (better match) from either search
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

        # Filter and sort by distance
        results = sorted(best.values(), key=lambda r: r["_distance"])
        results = [r for r in results if r["_distance"] <= threshold]

        return results[:k]

search_service = SearchService()
