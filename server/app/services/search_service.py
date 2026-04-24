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
        self.model = AutoModel.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
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

    def search(
        self,
        query: str,
        k: int = 20,
        threshold: float = 0.20,
        force_rerank: bool = False,
        boost_weight: float = 0.80,
        rerank_top: int = 5,
        rerank_size: int = 768,
    ) -> list[dict]:
        """SigLIP image-vector search with OWL-ViT detection re-ranking.

        The query is embedded using a 'a photo of {query}' prompt template for
        improved SigLIP accuracy. Top candidates are then re-ranked by
        OWL-ViT detection confidence on the same query phrase.

        force_rerank: if True, bypass the SigLIP-confident skip and run OWL-ViT
        on every top candidate. Used for attribute-heavy clauses where SigLIP
        is unreliable at binding attributes to objects.

        boost_weight: fraction of the gap (1 - final_sim) that a confident
        detection closes. Default 0.80; raise to ~0.95 for attribute clauses
        so OWL-ViT evidence dominates.
        """
        if self.table is None:
            self.refresh_table()
            if self.table is None:
                return []

        vector_query = f"a photo of {query.strip()}"
        query_vec = self.embed_text(vector_query)
        select_cols = ["photo_id", "photo_image_url", "video_url", "timestamp"]
        ranked_lists = []

        # Wider net (k * 2) so we have enough candidates before filtering via threshold.
        try:
            df = (
                self.table.search(query_vec, vector_column_name="vector")
                .metric("cosine")
                .limit(k * 2)
                .select(select_cols)
                .to_pandas()
            )
            # LanceDB auto-injects _distance after .search(); don't select it explicitly.
            ranked_lists.append(df)
        except Exception as e:
            print(f"Image vector search failed: {e}")

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

                if pid not in best_row:
                    best_row[pid] = row.to_dict()
                elif row_dist < float(best_row[pid].get("_distance", 1.0)):
                    best_row[pid]["_distance"] = row_dist

        if not rrf_scores:
            return []

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

        def _apply_det_boost(pid: str, score: float):
            best_row[pid]["_detection_score"] = score
            if score > 0.15:
                rrf_scores[pid] += (score * 10.0)
            elif score > 0.05:
                rrf_scores[pid] += (score * 2.0)

        # Knob 1: trim to top-N for OWL-ViT re-rank. On HF cpu-basic each
        # OWL-ViT forward pass is ~700-900ms at 768px; demo-mode uses N=3 @ 512px.
        top_candidates = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)[:rerank_top]

        from app.services.detection_service import detection_service
        import os
        from PIL import Image

        # Knob 2: skip re-rank on candidates where SigLIP is already confident
        # (rescaled >= 0.65, i.e. raw cosine sim >= ~0.13). Saves a full
        # OWL-ViT forward pass per skip.
        for pid in top_candidates:
            row = best_row[pid]
            row_dist = float(row.get("_distance", 1.0))
            if not force_rerank and rescale_dist(row_dist) >= 0.65:
                continue
            url = row.get("photo_image_url", "")
            if "/images/" not in url:
                continue
            file_name = url.split("/images/")[-1]
            local_path = os.path.join(os.getenv("DATA_DIR", "data"), file_name)
            if not os.path.exists(local_path):
                continue
            try:
                img = Image.open(local_path).convert("RGB")
                img.thumbnail((rerank_size, rerank_size))
                det = detection_service.detect(img, query)
                if det:
                    _apply_det_boost(pid, det.get("score", 0))
            except Exception:
                continue

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
            det_score = float(row.get("_detection_score", 0.0))

            # 1. Base score from SigLIP (cross-modal similarity, rescaled to 0..1)
            final_sim = rescale_dist(dist)

            # 2. OWL-ViT detection re-rank — close `boost_weight` of the gap weighted by confidence
            if det_score > 0.10:
                final_sim += (1.0 - final_sim) * (boost_weight * det_score)

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
