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
        """Score how well the stored description confirms the query object.

        Uses a two-tier approach:
          1. PRIMARY NOUN match: Does the description mention the core object?
             This is the decisive filter — pass or fail.
          2. ADJECTIVE match: Among confirmed objects, do the adjectives match?
             This is a ranking bonus, not a filter.

        Returns (noun_score, adjective_bonus) where:
          noun_score:       0.0 (no match) or 0.85–1.0 (confirmed)
          adjective_bonus:  0.0–0.10 (how many adjectives also match)
        """
        desc = description.lower()
        q = query.strip().lower()

        # ── Noun matching (the PRIMARY signal) ────────────────────────
        nouns = _extract_nouns(q)
        if not nouns:
            return (0.0, 0.0)

        # Check whole-word matches only ('dog' must not match 'hotdog')
        noun_hits = sum(1 for n in nouns if re.search(r'\b' + re.escape(n) + r'\b', desc))
        noun_ratio = noun_hits / len(nouns)

        if noun_ratio <= 0:
            return (0.0, 0.0)

        # Full phrase match is best
        if re.search(r'\b' + re.escape(q) + r'\b', desc):
            noun_score = 1.0
        elif noun_ratio >= 1.0:
            noun_score = 0.92       # all nouns present, different order
        else:
            noun_score = 0.85       # at least one core noun confirmed

        # ── Adjective matching (RANKING bonus only) ───────────────────
        adjectives = _extract_adjectives(q)
        if adjectives:
            adj_hits = sum(1 for a in adjectives
                           if re.search(r'\b' + re.escape(a) + r'\b', desc))
            adjective_bonus = (adj_hits / len(adjectives)) * 0.07
        else:
            adjective_bonus = 0.0

        return (noun_score, adjective_bonus)

    # ── Main search ────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 20, threshold: float = 0.20,
               deep_search: bool = False) -> list:
        """Description-first hybrid search.

        Architecture
        ------------
        The key insight: Florence-2 already identified every object at upload
        time.  The stored description IS the ground truth for object identity.

        Layer 1 — Retrieval (SigLIP ANN + BM25):
          Fast candidate retrieval (~60 items, <1s).

        Layer 2 — Object Identification (Description Match):
          PRIMARY SIGNAL. If the core noun from the query appears in the
          stored description, the image is confirmed.  Adjectives (brown,
          small) are used as a ranking bonus, not a filter.

        Layer 3 — Ranking:
          Confirmed images:   0.85–0.99 (noun match + adj bonus + rank bonus)
          Unconfirmed images: 0.05–0.25 (visual rank only, always below threshold)

        Layer 4 — Deep Search (optional, ~30s):
          Florence-2 phrase-grounding on top 8 CONFIRMED results.
          Only boosts already-confirmed results — never promotes unconfirmed ones.

        Score ranges (with default 0.80 threshold):
          0.85–0.99  =  Description confirms the object
          0.05–0.25  =  Visually similar but description doesn't match → filtered
        """
        if self.table is None:
            self.refresh_table()
            if self.table is None:
                return []

        vector_query = f"a photo of {query.strip()}"
        query_vec = self.embed_text(vector_query)
        select_cols = [
            "photo_id", "photo_image_url", "video_url",
            "timestamp", "description", "_distance"
        ]
        ranked_lists = []

        # ── Layer 1a: Image vector ANN ───────────────────────────────────
        try:
            df = (self.table.search(query_vec, vector_column_name="vector")
                  .metric("cosine").limit(k * 3).select(select_cols).to_pandas())
            ranked_lists.append(df)
        except Exception as e:
            print(f"Image ANN failed: {e}")

        # ── Layer 1b: Caption vector ANN ─────────────────────────────────
        try:
            df = (self.table.search(query_vec, vector_column_name="caption_vector")
                  .metric("cosine").limit(k * 3).select(select_cols).to_pandas())
            ranked_lists.append(df)
        except Exception as e:
            print(f"Caption ANN failed: {e}")

        # ── Layer 1c: BM25 full-text ─────────────────────────────────────
        bm25_pids: set = set()
        try:
            fts_df = (self.table.search(query, query_type="fts")
                      .limit(k * 3)
                      .select(["photo_id", "photo_image_url", "video_url",
                               "timestamp", "description"])
                      .to_pandas())
            fts_df["_distance"] = 1.0
            ranked_lists.append(fts_df)
            bm25_pids = set(str(p) for p in fts_df["photo_id"].tolist())
        except Exception as e:
            print(f"BM25 skipped: {e}")

        # ── RRF Fusion ───────────────────────────────────────────────────
        RRF_K = 60
        rrf_scores: dict = {}
        best_row: dict = {}

        for result_df in ranked_lists:
            if result_df is None or result_df.empty:
                continue
            for rank, (_, row) in enumerate(result_df.iterrows()):
                pid = str(row["photo_id"])
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
                if pid not in best_row:
                    best_row[pid] = row.to_dict()

        if not rrf_scores:
            return []

        # ── Layer 2 + 3: Description scoring ─────────────────────────────
        sorted_pids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)
        total = len(sorted_pids)

        for rank_pos, pid in enumerate(sorted_pids):
            row = best_row[pid]
            desc = str(row.get("description", ""))

            noun_score, adj_bonus = self._description_match_score(query, desc)

            # BM25 floor: only if BM25 matched AND the core noun appears
            # in the description (prevents "brown horse" from being confirmed
            # for query "brown dog")
            if noun_score == 0.0 and pid in bm25_pids:
                nouns = _extract_nouns(query)
                desc_lower = desc.lower()
                if nouns and any(re.search(r'\b' + re.escape(n) + r'\b', desc_lower)
                                 for n in nouns):
                    noun_score = 0.85

            # Visual Similarity from SigLIP (1.0 - distance)
            # Typically ranges from 0.15 to 0.35, but can be higher.
            visual_sim = max(0.0, 1.0 - float(row.get("_distance", 1.0)))

            if noun_score > 0:
                # CONFIRMED: core noun is in description
                # Base score 0.80 + Adjective bonus (up to 0.07) + Visual similarity (up to ~0.15)
                # This guarantees confirmed items pass the 0.80 threshold, but provides wide
                # score variation based on actual visual match (e.g., "black dog" vs "dog" color match)
                combined = 0.80 + adj_bonus + (visual_sim * 0.5)
            else:
                # NOT CONFIRMED: Rank by visual similarity alone, suppressed by 0.50 threshold penalty
                combined = visual_sim * 0.5

            best_row[pid]["_noun_score"] = noun_score
            best_row[pid]["_combined_score"] = min(0.99, combined)

        # ── Layer 4: Florence-2 Deep Search (optional, top 8 confirmed) ──
        DEEP_LIMIT = 8
        if deep_search:
            from app.services.caption_service import caption_service

            # ONLY run on CONFIRMED results (noun_score > 0)
            confirmed = [p for p in sorted_pids
                         if best_row[p].get("_noun_score", 0) > 0][:DEEP_LIMIT]

            print(f"[Deep Search] Florence-2 grounding on {len(confirmed)} confirmed candidates…")
            for pid in confirmed:
                row = best_row[pid]
                url = row.get("photo_image_url", "")
                if "/images/" not in url:
                    continue
                file_name = url.split("/images/")[-1]
                local_path = os.path.join("data", file_name)
                if not os.path.exists(local_path):
                    continue
                try:
                    img = Image.open(local_path).convert("RGB")
                    img.thumbnail((800, 800))
                    grounding = caption_service.ground_phrase(img, query)
                    best_row[pid]["_verified"] = grounding["found"]
                    best_row[pid]["_f2_score"] = grounding["score"]
                    best_row[pid]["_f2_box"] = grounding["box"]
                    if grounding["found"]:
                        # Small bonus to already-confirmed result — never
                        # exceeds current score + 0.04 (prevents over-boosting)
                        current = best_row[pid]["_combined_score"]
                        best_row[pid]["_combined_score"] = min(0.99,
                            current + 0.03)
                    print(f"  [{pid}] found={grounding['found']} "
                          f"score={grounding['score']:.3f}")
                except Exception as e:
                    print(f"  Florence-2 error for {pid}: {e}")

        # ── Build final result list ──────────────────────────────────────
        seen_videos: set = set()
        results = []

        final_order = sorted(
            best_row.keys(),
            key=lambda p: best_row[p].get("_combined_score", 0),
            reverse=True
        )

        for pid in final_order:
            row = best_row[pid].copy()
            v_url = str(row.get("video_url") or "").strip()

            if v_url and v_url in seen_videos:
                continue
            if v_url:
                seen_videos.add(v_url)

            final_sim = row.get("_combined_score", 0.0)
            if final_sim < threshold:
                continue

            row["similarity_score"] = final_sim
            row["verified"] = bool(row.get("_verified", False))
            row["florence_box"] = row.get("_f2_box", None)
            results.append(row)

            if len(results) >= k:
                break

        return results


search_service = SearchService()
