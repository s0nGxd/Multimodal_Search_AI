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

        vector_query = f"a photo of {query.strip()}"
        query_vec = self.embed_text(vector_query)
        select_cols = ["photo_id", "photo_image_url", "video_url", "timestamp", "description", "_distance"]
        ranked_lists = []

        # We query wider nets (k * 2) so we have enough candidates before filtering via threshold at the end
        try:
            df = (
                self.table.search(query_vec, vector_column_name="vector")
                .metric("cosine")
                .limit(k * 2)
                .select(select_cols)
                .to_pandas()
            )
            ranked_lists.append(df)
        except Exception as e:
            print(f"Image vector search failed: {e}")

        try:
            df = (
                self.table.search(query_vec, vector_column_name="caption_vector")
                .metric("cosine")
                .limit(k * 2)
                .select(select_cols)
                .to_pandas()
            )
            ranked_lists.append(df)
        except Exception as e:
            print(f"Caption vector search failed: {e}")

        try:
            fts_df = (
                self.table.search(query, query_type="fts")
                .limit(k * 2)
                .select(["photo_id", "photo_image_url", "video_url", "timestamp", "description"])
                .to_pandas()
            )
            fts_df["_distance"] = 1.0
            fts_df["_bm25_found"] = True
            ranked_lists.append(fts_df)
        except Exception as e:
            print(f"BM25 search skipped: {e}")

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
                is_bm25 = bool(row.get("_bm25_found", False))
                
                if pid not in best_row:
                    best_row[pid] = row.to_dict()
                else:
                    if row_dist < float(best_row[pid].get("_distance", 1.0)):
                        best_row[pid]["_distance"] = row_dist
                    if is_bm25:
                        best_row[pid]["_bm25_found"] = True

        if not rrf_scores:
            return []

        top_candidates = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)[:15]
        
        from app.services.detection_service import detection_service
        import os
        from PIL import Image
        
        for pid in top_candidates:
            row = best_row[pid]
            url = row.get("photo_image_url", "")
            if "/images/" in url:
                file_name = url.split("/images/")[-1]
                local_path = os.path.join("data", file_name)
                if os.path.exists(local_path):
                    try:
                        img = Image.open(local_path).convert("RGB")
                        img.thumbnail((768, 768)) # aggressive optimization for huge memory bounds
                        det = detection_service.detect(img, query)
                        if det:
                            score = det.get("score", 0)
                            if score > 0.15:
                                rrf_scores[pid] += (score * 10.0)
                            elif score > 0.05:
                                rrf_scores[pid] += (score * 2.0)
                    except Exception as e:
                        pass 

        sorted_pids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)

        SIM_FLOOR = 0.15
        SIM_CEIL = 1.00

        def rescale_dist(dist: float) -> float:
            raw_sim = 1.0 - dist
            scaled = (raw_sim - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR)
            return max(0.0, min(1.0, scaled))

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
            is_bm25 = row.get("_bm25_found", False)
            
            if dist > 0.99 and is_bm25:
                base_sim = 0.70  # Standard 70% confidence for exact text matches
            else:
                base_sim = rescale_dist(dist)
                if is_bm25:
                    base_sim += 0.20 # Bump visual matches by 20% if text explicitly mentions it
            
            if rrf_scores[pid] >= 1.0:
                # Add 30% absolute confidence flat bump for verified OWL-ViT physical objects
                base_sim += 0.30
                
            final_sim = min(1.0, max(0.0, base_sim))
            
            # Enforce the user interface threshold slider parameter!
            if final_sim < threshold:
                continue
                
            row["_rrf_score"] = rrf_scores[pid]
            row["similarity_score"] = final_sim
            
            results.append(row)
            if len(results) >= k:
                break
                
        return results

search_service = SearchService()
