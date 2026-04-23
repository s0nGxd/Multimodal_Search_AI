import os
from collections import OrderedDict
import re
import lancedb
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoProcessor
from PIL import Image


# ── Words to strip when extracting the PRIMARY NOUN from a query ──────────
# These are adjectives, colours, sizes, articles, prepositions, and filler
# words that describe an object but are NOT the object itself.
_MODIFIERS = {
    # colours
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange',
    'purple', 'pink', 'grey', 'gray', 'golden', 'silver', 'dark', 'light',
    'bright', 'pale', 'beige', 'tan', 'cream',
    # size / age / shape
    'large', 'small', 'big', 'tiny', 'tall', 'short', 'long', 'old', 'young',
    'round', 'thin', 'thick', 'wide', 'narrow', 'huge', 'little', 'massive',
    # texture / quality
    'fluffy', 'furry', 'smooth', 'rough', 'shiny', 'soft', 'hard',
    # filler / articles / prepositions
    'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'for',
    'is', 'to', 'by', 'from', 'its', 'this', 'that',
    # common query filler words
    'color', 'colour', 'colored', 'coloured', 'looking', 'like', 'type',
    'kind', 'style', 'very', 'really', 'quite', 'pretty', 'beautiful',
}


def _extract_nouns(phrase: str) -> list[str]:
    """Extract meaningful content nouns from a query, stripping modifiers.

    'brown color dog running' → ['dog', 'running']
    'small black cat'         → ['cat']
    'woman in red jacket'     → ['woman', 'jacket']
    """
    words = [w.lower().strip(".,!?") for w in phrase.split() if len(w) > 1]
    nouns = [w for w in words if w not in _MODIFIERS]
    return nouns if nouns else words


def _extract_adjectives(phrase: str) -> list[str]:
    """Extract modifier/adjective words from a query.

    'brown color dog' → ['brown']
    'small black cat' → ['small', 'black']
    """
    words = [w.lower().strip(".,!?") for w in phrase.split() if len(w) > 1]
    # Only return words that ARE in the modifier set and are visually meaningful
    visual_modifiers = {
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange',
        'purple', 'pink', 'grey', 'gray', 'golden', 'silver', 'dark', 'light',
        'large', 'small', 'big', 'tiny', 'tall', 'short', 'long', 'old', 'young',
        'fluffy', 'furry', 'smooth', 'shiny',
    }
    return [w for w in words if w in visual_modifiers]


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
        # SigLIP Base uses 768 dimension
        self.embed_dim = int(os.getenv("EMBED_DIM", "768"))
        self.model_name = os.getenv("EMBED_MODEL", "google/siglip-base-patch16-224")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._text_cache = OrderedDict()
        self._text_cache_max = 128

        print(f"Loading vision-language model: {self.model_name} on {self.device}")
        self.model = AutoModel.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()

        self.db = lancedb.connect(self.db_uri)
        try:
            self.table = self.db.open_table("images")
        except Exception:
            print("Table 'images' not found. It will need to be created via ingestion.")
            self.table = None

    def refresh_table(self):
        try:
            self.table = self.db.open_table("images")
        except Exception:
            self.table = None

    # ── Embedding helpers ──────────────────────────────────────────────────

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
    def embed_images_batch(self, images: list, batch_size: int = 16) -> list:
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

    # ── Description-based object scoring ───────────────────────────────────

    @staticmethod
    def _description_match_score(query: str, description: str) -> tuple[float, float]:
        """Score how well the stored description confirms the query object."""
        desc = description.lower()
        q = query.strip().lower()

        # ── Noun matching (the PRIMARY signal) ────────────────────────
        nouns = _extract_nouns(q)
        if not nouns:
            return (0.0, 0.0)

        # Check whole-word matches only
        noun_hits = sum(1 for n in nouns if re.search(r'\b' + re.escape(n) + r'\b', desc))
        noun_ratio = noun_hits / len(nouns)

        if noun_ratio <= 0:
            return (0.0, 0.0)

        # ── Adjective matching (RANKING & PENALTY logic) ──────────────
        adjectives = _extract_adjectives(q)
        adjective_bonus = 0.0
        
        if adjectives:
            # Check for matches
            adj_hits = sum(1 for a in adjectives
                           if re.search(r'\b' + re.escape(a) + r'\b', desc))
            
            # If some adjectives matched, calculate bonus
            if adj_hits > 0:
                adjective_bonus = (adj_hits / len(adjectives)) * 0.08
            
            # ── COLOR CONFLICT PENALTY ──
            colors = {
                'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 
                'orange', 'purple', 'pink', 'grey', 'gray', 'golden', 'silver'
            }
            query_colors = set(adjectives) & colors
            if query_colors:
                desc_words = set(re.findall(r'\b\w+\b', desc))
                desc_colors = desc_words & colors
                
                if query_colors & desc_colors:
                    # Great! The requested color is mentioned.
                    noun_base = 0.85
                elif desc_colors:
                    # A DIFFERENT color is mentioned, and ours is NOT. Conflict!
                    # Return 0.0 so we can fallback to region_desc scoring.
                    return (0.0, 0.0)
                else:
                    # No color mentioned at all.
                    noun_base = 0.70
            else:
                noun_base = 0.85
        else:
            noun_base = 0.85

        # Final noun_score calculation
        if re.search(r'\b' + re.escape(q) + r'\b', desc):
            noun_score = noun_base + 0.14 # e.g. 0.99
        elif noun_ratio >= 1.0:
            noun_score = noun_base + 0.07 # e.g. 0.92
        else:
            noun_score = noun_base        # e.g. 0.85

        return (noun_score, adjective_bonus)

    # ── Main search ────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 20, threshold: float = 0.20,
               deep_search: bool = False) -> list:
        """Description-first hybrid search."""
        import json
        import datetime
        
        # ── Setup Debug Logging ──────────────────────────────────────
        log_file = "search_debug.log"
        def log_debug(msg):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {msg}\n")

        log_debug(f"\n{'='*80}\nNEW SEARCH REQUEST: '{query}'\n{'='*80}")
        query_nouns = _extract_nouns(query)
        query_adjectives = _extract_adjectives(query)
        log_debug(f"Extracted Nouns: {query_nouns}")
        log_debug(f"Extracted Adjectives: {query_adjectives}")

        if self.table is None:
            self.refresh_table()
            if self.table is None:
                log_debug("ERROR: LanceDB table not found.")
                return []

        vector_query = f"a photo of {query.strip()}"
        query_vec = self.embed_text(vector_query)
        select_cols = [
            "photo_id", "photo_image_url", "video_url",
            "timestamp", "description", "_distance",
            "is_region", "parent_photo_id", "bbox", "region_label"
        ]
        fts_select_cols = [
            "photo_id", "photo_image_url", "video_url",
            "timestamp", "description",
            "is_region", "parent_photo_id", "bbox", "region_label"
        ]
        ranked_lists = []

        # ── Layer 1a: Image vector ANN ───────────────────────────────────
        try:
            df = (self.table.search(query_vec, vector_column_name="vector")
                  .metric("cosine").limit(k * 3).select(select_cols).to_pandas())
            ranked_lists.append(df)
            log_debug(f"Layer 1a (Image ANN): Retrieved {len(df)} candidates.")
        except Exception as e:
            log_debug(f"Layer 1a (Image ANN) FAILED: {e}")

        # ── Layer 1b: Caption vector ANN ─────────────────────────────────
        try:
            df = (self.table.search(query_vec, vector_column_name="caption_vector")
                  .metric("cosine").limit(k * 3).select(select_cols).to_pandas())
            ranked_lists.append(df)
            log_debug(f"Layer 1b (Caption ANN): Retrieved {len(df)} candidates.")
        except Exception as e:
            log_debug(f"Layer 1b (Caption ANN) FAILED: {e}")

        # ── Layer 1c: BM25 full-text ─────────────────────────────────────
        bm25_pids: set = set()
        try:
            fts_df = (self.table.search(query, query_type="fts")
                      .limit(k * 3)
                      .select(fts_select_cols)
                      .to_pandas())
            fts_df["_distance"] = 1.0
            ranked_lists.append(fts_df)
            bm25_pids = set(str(p) for p in fts_df["photo_id"].tolist())
            log_debug(f"Layer 1c (BM25): Retrieved {len(fts_df)} candidates.")
        except Exception as e:
            log_debug(f"Layer 1c (BM25) skipped: {e}")

        # ── RRF Fusion & Region Deduplication ────────────────────────────
        import re as _re
        
        RRF_K = 60
        rrf_scores: dict = {}
        best_row: dict = {}
        parent_region_info: dict = {}  # parent_pid -> {"bbox": ..., "description": ..., "score": ...}

        for i, result_df in enumerate(ranked_lists):
            if result_df is None or result_df.empty:
                continue
            for rank, (_, row) in enumerate(result_df.iterrows()):
                is_region = row.get("is_region", False)
                rrf_contribution = 1.0 / (RRF_K + rank + 1)
                
                if is_region and row.get("parent_photo_id"):
                    parent_pid = str(row["parent_photo_id"])
                    region_desc = str(row.get("description", "")).lower()
                    
                    noun_hit = any(
                        _re.search(r'\b' + _re.escape(n) + r'\b', region_desc)
                        for n in query_nouns
                    ) if query_nouns else False
                    
                    if noun_hit:
                        rrf_scores[parent_pid] = rrf_scores.get(parent_pid, 0.0) + rrf_contribution
                        region_noun, region_adj = self._description_match_score(query, region_desc)
                        region_match_quality = region_noun + region_adj
                        
                        prev_quality = parent_region_info.get(parent_pid, {}).get("match_quality", -1.0)
                        if region_match_quality > prev_quality:
                            bbox_str = row.get("bbox", "")
                            parent_region_info[parent_pid] = {
                                "bbox": json.loads(bbox_str) if bbox_str else None,
                                "description": row.get("region_label", row.get("description", "")),
                                "match_quality": region_match_quality
                            }
                else:
                    pid = str(row["photo_id"])
                    rrf_scores[pid] = rrf_scores.get(pid, 0.0) + rrf_contribution
                    if pid not in best_row:
                        best_row[pid] = row.to_dict()
        
        log_debug(f"RRF Fusion: {len(rrf_scores)} unique frames ranked.")
        
        for parent_pid, region_info in parent_region_info.items():
            if parent_pid not in best_row:
                try:
                    parent_df = self.table.search().where(f"photo_id = '{parent_pid}'").limit(1).to_pandas()
                    if not parent_df.empty:
                        best_row[parent_pid] = parent_df.iloc[0].to_dict()
                        best_row[parent_pid]["_distance"] = 1.0 
                    else:
                        continue
                except Exception:
                    continue

            if region_info.get("bbox"):
                best_row[parent_pid]["_f2_box"] = region_info["bbox"]
            if region_info.get("description"):
                best_row[parent_pid]["_region_description"] = region_info["description"]

        if not rrf_scores:
            log_debug("No results found.")
            return []

        # ── Layer 2 + 3: Description scoring ─────────────────────────────
        sorted_pids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)
        log_debug("\nSCORING BREAKDOWN (Top 10):")

        for rank_pos, pid in enumerate(sorted_pids):
            if pid not in best_row:
                continue
            row = best_row[pid]
            desc = str(row.get("description", ""))
            region_desc = str(row.get("_region_description", ""))

            noun_score, adj_bonus = self._description_match_score(query, desc)
            if noun_score == 0.0 and region_desc:
                noun_score, adj_bonus = self._description_match_score(query, region_desc)

            if noun_score == 0.0 and pid in bm25_pids:
                desc_lower = desc.lower() + " " + region_desc.lower()
                if query_nouns and any(re.search(r'\b' + re.escape(n) + r'\b', desc_lower)
                                 for n in query_nouns):
                    noun_score = 0.85

            visual_sim = max(0.0, 1.0 - float(row.get("_distance", 1.0)))

            if noun_score > 0:
                combined = noun_score + adj_bonus + (visual_sim * 0.15)
            else:
                combined = visual_sim * 0.25

            best_row[pid]["_noun_score"] = noun_score
            best_row[pid]["_combined_score"] = min(0.99, combined)
            
            if rank_pos < 10:
                log_debug(f"[{rank_pos+1}] PID: {pid}")
                log_debug(f"    Global Desc: '{desc[:60]}...'")
                log_debug(f"    Region Desc: '{region_desc[:60]}...'")
                log_debug(f"    Scores: Noun={noun_score:.2f}, AdjBonus={adj_bonus:.2f}, VisualSim={visual_sim:.2f}")
                log_debug(f"    FINAL SCORE: {combined:.4f}")

        # ── Layer 4: Florence-2 Deep Search (optional) ──────────────────
        DEEP_LIMIT = 8
        if deep_search:
            from app.services.caption_service import caption_service
            confirmed = [p for p in sorted_pids
                         if p in best_row and best_row[p].get("_noun_score", 0) > 0][:DEEP_LIMIT]

            log_debug(f"\n[Deep Search] Grounding on {len(confirmed)} candidates...")
            for pid in confirmed:
                row = best_row[pid]
                url = row.get("photo_image_url", "")
                if "/images/" not in url: continue
                local_path = os.path.join("data", url.split("/images/")[-1])
                if not os.path.exists(local_path): continue
                try:
                    img = Image.open(local_path).convert("RGB")
                    img.thumbnail((800, 800))
                    grounding = caption_service.ground_phrase(img, query)
                    best_row[pid]["_verified"] = grounding["found"]
                    
                    # CRITICAL FIX: Only use Florence-2 box if we don't already have one.
                    # This prevents DeepSearch from overwriting perfect YOLO regions.
                    if grounding["found"]:
                        if not best_row[pid].get("_f2_box"):
                            best_row[pid]["_f2_box"] = grounding["box"]
                        best_row[pid]["_combined_score"] = min(0.99, best_row[pid]["_combined_score"] + 0.03)
                        
                    log_debug(f"    PID {pid}: found={grounding['found']}, box={grounding['box']}")
                except Exception as e:
                    log_debug(f"    PID {pid}: Deep Search ERROR: {e}")

        # ── Build final result list ──────────────────────────────────────
        seen_videos: set = set()
        results = []
        final_order = sorted(best_row.keys(), key=lambda p: best_row[p].get("_combined_score", 0), reverse=True)

        log_debug("\nFINAL TOP 5 RESULTS:")
        for pid in final_order:
            row = best_row[pid].copy()
            v_url = str(row.get("video_url") or "").strip()
            if v_url and v_url in seen_videos: continue
            if v_url: seen_videos.add(v_url)

            final_sim = row.get("_combined_score", 0.0)
            if final_sim < threshold: continue

            row["similarity_score"] = final_sim
            row["verified"] = bool(row.get("_verified", False))
            row["florence_box"] = row.get("_f2_box", None)
            results.append(row)
            
            if len(results) <= 5:
                log_debug(f"    - {pid}: Score {final_sim:.4f} ({row.get('photo_image_url')})")

            if len(results) >= k: break

        log_debug(f"Search complete. Returning {len(results)} results.\n")
        return results


search_service = SearchService()
