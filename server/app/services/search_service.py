import os
from collections import OrderedDict
import lancedb
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoProcessor
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
        self.model_name = os.getenv("EMBED_MODEL", "google/siglip-base-patch16-256")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Text embedding cache (LRU, max 128 entries)
        self._text_cache = OrderedDict()
        self._text_cache_max = 128

        # Load Model (AutoModel supports CLIP, SigLIP, and other vision-language models)
        print(f"Loading vision-language model: {self.model_name} on {self.device}")
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
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

    def search(self, query: str, k: int = 20, threshold: float = 0.75):
        """Hybrid search using 3 channels fused via Reciprocal Rank Fusion (RRF).

        Channels:
          1. Image vector search  — visual similarity (SigLIP image embedding)
          2. Caption vector search — semantic text match (SigLIP text embedding vs caption)
          3. BM25 full-text search — exact keyword match on the description field

        The vector channels use a 'a photo of {query}' prompt template for improved
        SigLIP accuracy. BM25 uses the raw query for exact keyword matching.

        Results are merged with RRF. Each result carries its _rrf_score so the router
        can display it as a normalized, intuitive percentage.
        """
        if self.table is None:
            self.refresh_table()
            if self.table is None:
                return []

        # Apply SigLIP prompt template for vector search only — improves zero-shot accuracy
        vector_query = f"a photo of {query.strip()}"
        query_vec = self.embed_text(vector_query)
        select_cols = ["photo_id", "photo_image_url", "description", "_distance"]
        ranked_lists: list[pd.DataFrame] = []

        # --- Channel 1: Image vector search (visual similarity) ---
        try:
            df = (
                self.table.search(query_vec, vector_column_name="vector")
                .metric("cosine")
                .limit(k)
                .select(select_cols)
                .to_pandas()
            )
            df = df[df["_distance"] <= threshold]
            ranked_lists.append(df)
        except Exception as e:
            print(f"Image vector search failed: {e}")
            ranked_lists.append(pd.DataFrame())

        # --- Channel 2: Caption vector search (semantic text match) ---
        try:
            df = (
                self.table.search(query_vec, vector_column_name="caption_vector")
                .metric("cosine")
                .limit(k)
                .select(select_cols)
                .to_pandas()
            )
            df = df[df["_distance"] <= threshold]
            ranked_lists.append(df)
        except Exception as e:
            print(f"Caption vector search failed: {e}")
            ranked_lists.append(pd.DataFrame())

        # --- Channel 3: BM25 full-text search (raw query, exact keyword match) ---
        try:
            fts_df = (
                self.table.search(query, query_type="fts")  # raw query — no template
                .limit(k)
                .select(["photo_id", "photo_image_url", "description"])
                .to_pandas()
            )
            fts_df["_distance"] = 1.0  # placeholder only; not used for scoring
            ranked_lists.append(fts_df)
        except Exception as e:
            print(f"BM25 search skipped (index may not exist yet): {e}")

        # --- Reciprocal Rank Fusion ---
        # Each channel contributes 1/(RRF_K + rank) to a shared score.
        # Higher combined score = better overall match across all channels.
        RRF_K = 60
        rrf_scores: dict[str, float] = {}
        best_row: dict[str, dict] = {}

        for result_df in ranked_lists:
            if result_df is None or result_df.empty:
                continue
            for rank, (_, row) in enumerate(result_df.iterrows()):
                pid = row["photo_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
                row_dist = float(row.get("_distance", 1.0))
                if pid not in best_row or row_dist < float(best_row[pid].get("_distance", 1.0)):
                    best_row[pid] = row.to_dict()

        if not rrf_scores:
            return []

        # Sort descending by RRF score (higher = better combined match)
        sorted_pids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)
        max_rrf = rrf_scores[sorted_pids[0]]

        results = []
        for pid in sorted_pids[:k]:
            row = best_row[pid].copy()
            # Attach normalized RRF score: top result = 1.0, others proportionally lower.
            # This gives a score that reflects multi-channel relevance, not raw vector distance.
            row["_rrf_score"] = rrf_scores[pid] / max_rrf
            results.append(row)
        return results

search_service = SearchService()
