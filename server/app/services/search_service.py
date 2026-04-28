import os
from collections import OrderedDict
import lancedb
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoProcessor
from PIL import Image
from typing import Optional

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

        self.model = None
        self.processor = None
        self.load_model()

        # Connect to DB
        self.db = lancedb.connect(self.db_uri)
        try:
            self.table = self.db.open_table("images")
        except:
            print("Table 'images' not found. It will need to be created via ingestion.")
            self.table = None

    def load_model(self):
        if self.model is not None:
            return
        print(f"Loading vision-language model: {self.model_name} on {self.device}")
        self.model = AutoModel.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()

    def unload_model(self):
        print(f"Unloading vision-language model: {self.model_name}")
        self.model = None
        self.processor = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def refresh_table(self):
        """Re-opens the table and verifies health."""
        try:
            self.table = self.db.open_table("images")
            # Verify health with a simple count
            _ = self.table.count_rows()
        except Exception as e:
            if self.table is not None:
                print(f"Warning: Database table 'images' appears corrupted or missing: {e}")
            self.table = None

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        self.load_model()
        cache_key = text.strip().lower()
        if cache_key in self._text_cache:
            self._text_cache.move_to_end(cache_key)
            return self._text_cache[cache_key]

        # SigLIP expects padding='max_length' (fixed 64 tokens). Using padding=True
        # gives wrong embeddings for single queries.
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(self.device)
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
        self.load_model()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)

        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype("float32")[0]

    @torch.no_grad()
    def embed_images_batch(self, images: list[Image.Image], batch_size: int = 16) -> list[np.ndarray]:
        self.load_model()
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

    def search(
        self,
        query: str,
        k: int = 20,
        threshold: float = 0.20,
        pre_filter: Optional[str] = None,
    ) -> list[dict]:
        """SigLIP image-vector search with optional LanceDB SQL pre-filtering.

        pre_filter: A SQL WHERE clause (e.g. "objects_json LIKE '%red%'")
        """
        if self.table is None:
            self.refresh_table()
            if self.table is None:
                return []

        vector_query = f"a photo of {query.strip()}"
        query_vec = self.embed_text(vector_query)
        select_cols = ["photo_id", "photo_image_url", "video_url", "timestamp", "objects_json"]
        
        try:
            search_query = self.table.search(query_vec, vector_column_name="vector").metric("cosine")
            
            if pre_filter:
                print(f"DEBUG: Applying pre-filter: {pre_filter}")
                search_query = search_query.where(pre_filter, pre_filter_mode="compat")
                
            df = (
                search_query
                .limit(k * 3) # Wider net since we might filter more
                .select(select_cols)
                .to_pandas()
            )
        except Exception as e:
            print(f"Image vector search failed: {e}")
            return []

        if df.empty:
            return []

        RRF_K = 60
        rrf_scores: dict[str, float] = {}
        best_row: dict[str, dict] = {}

        for rank, (_, row) in enumerate(df.iterrows()):
            pid = row["photo_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
            row_dist = float(row.get("_distance", 1.0))
            if pid not in best_row:
                best_row[pid] = row.to_dict()
            elif row_dist < float(best_row[pid].get("_distance", 1.0)):
                best_row[pid]["_distance"] = row_dist

        # Calibrated for pure SigLIP cross-modal (no caption_vector).
        # SigLIP single-query text-image cosine sim on this model+dataset
        # tops out around 0.06-0.10 for genuine matches. SIM_CEIL=0.10 maps
        # a strong bare-SigLIP match to ~100% before re-rank. Detection
        # boost still exists but is no longer load-bearing for perceived
        # confidence on hard queries where OWL-ViT gives weak signals.
        SIM_FLOOR = 0.00
        SIM_CEIL = 0.10

        def rescale_dist(dist: float) -> float:
            raw_sim = 1.0 - dist
            scaled = (raw_sim - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR)
            return max(0.0, min(1.0, scaled))

        sorted_pids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)

        seen_videos = set()
        results = []
        for pid in sorted_pids:
            row = best_row[pid].copy()
            v_url = row.get("video_url", "")

            if v_url and v_url in seen_videos:
                continue
            if v_url:
                seen_videos.add(v_url)

            dist = float(row.get("_distance", 1.0))

            # Base score from SigLIP (cross-modal similarity, rescaled to 0..1)
            final_sim = rescale_dist(dist)
            final_sim = min(0.999, final_sim)

            if final_sim < threshold:
                continue

            # Surface the matched frame's timestamp as best_timestamp for video results.
            # Photos (no video_url) carry best_timestamp=None.
            frame_ts = row.get("timestamp")
            if v_url and frame_ts is not None and pd.notna(frame_ts):
                row["best_timestamp"] = float(frame_ts)
            else:
                row["best_timestamp"] = None

            row["_rrf_score"] = rrf_scores[pid]
            row["similarity_score"] = final_sim

            results.append(row)
            if len(results) >= k:
                break

        return results

search_service = SearchService()
