import os
import time
import pandas as pd
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.search_service import search_service

# Demo-mode clause cache: phrase -> (expires_at, rows). 60s TTL, keyed on the
# exact search phrase we pass to search_service so repeat clauses across
# queries ("white van", then "white van and road") reuse the OWL-ViT work.
_CLAUSE_CACHE: "OrderedDict[str, tuple[float, list[dict]]]" = OrderedDict()
_CLAUSE_CACHE_MAX = 64
_CLAUSE_CACHE_TTL = 60.0
_CLAUSE_POOL = ThreadPoolExecutor(max_workers=4)


def _cache_get(key: str):
    now = time.time()
    entry = _CLAUSE_CACHE.get(key)
    if not entry:
        return None
    expires, rows = entry
    if expires < now:
        _CLAUSE_CACHE.pop(key, None)
        return None
    _CLAUSE_CACHE.move_to_end(key)
    return rows


def _cache_put(key: str, rows: list[dict]):
    _CLAUSE_CACHE[key] = (time.time() + _CLAUSE_CACHE_TTL, rows)
    _CLAUSE_CACHE.move_to_end(key)
    while len(_CLAUSE_CACHE) > _CLAUSE_CACHE_MAX:
        _CLAUSE_CACHE.popitem(last=False)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 20
    threshold: Optional[float] = 0.20  # Min Similarity (0.0 - 1.0)

class SearchClause(BaseModel):
    object: str
    attributes: List[str] = []
    negated: bool = False

class SearchPlan(BaseModel):
    mode: str  # "AND" | "OR" | "SINGLE"
    clauses: List[SearchClause]

class ComplexSearchRequest(BaseModel):
    plan: SearchPlan
    k: Optional[int] = 20
    threshold: Optional[float] = 0.20

class SearchResult(BaseModel):
    photo_id: str
    photo_image_url: str
    video_url: Optional[str] = None
    timestamp: Optional[float] = None
    best_timestamp: Optional[float] = None
    description: Optional[str] = None
    score: float

@router.get("/video/frames")
def get_video_frames(url: str):
    try:
        if search_service.table is None:
            return []
            
        if "/images/" in url:
            url = f"/images/{url.split('/images/')[-1]}"
            
        df = search_service.table.to_pandas()
        video_df = df[df["video_url"] == url]
        
        frames = []
        for _, row in video_df.iterrows():
            frames.append({
                "timestamp": float(row.get("timestamp", 0.0)) if pd.notna(row.get("timestamp")) else 0.0,
                "description": row.get("description", "")
            })
            
        frames.sort(key=lambda x: x["timestamp"])
        return frames
    except Exception as e:
        print(f"Get video frames error: {e}")
        return []

@router.get("/images/all")
def list_all_images():
    try:
        if search_service.table is None:
            search_service.refresh_table()
            if search_service.table is None:
                return []
        df = search_service.table.to_pandas()
        if df.empty:
            return []
        # Project down to just the columns we render; the vector column rides
        # along in-memory but isn't shuffled through the response.
        keep_cols = [c for c in ("photo_id", "photo_image_url", "video_url", "timestamp") if c in df.columns]
        df = df[keep_cols].copy()

        # Vectorized URL rewriting: prefix BACKEND_URL onto /images/* URLs.
        photo_urls = df["photo_image_url"].astype(str)
        df["photo_image_url"] = photo_urls.where(
            ~photo_urls.str.startswith("/images/"),
            BACKEND_URL + photo_urls,
        )
        video_urls = df["video_url"].fillna("").astype(str)
        df["video_url"] = video_urls.where(
            ~video_urls.str.startswith("/images/"),
            BACKEND_URL + video_urls,
        )

        # Dedup videos (one card per video), keep all standalone photos.
        # Single images have video_url == "" (or None/NaN)
        is_video = df["video_url"].fillna("").str.len().gt(0)
        videos = df[is_video].drop_duplicates(subset=["video_url"], keep="first")
        photos = df[~is_video]
        combined = pd.concat([photos, videos], ignore_index=True)

        results = []
        for row in combined.itertuples(index=False):
            v_url = row.video_url if row.video_url else None
            ts = row.timestamp
            results.append({
                "photo_id": row.photo_id,
                "photo_image_url": row.photo_image_url,
                "video_url": v_url,
                "timestamp": float(ts) if pd.notna(ts) else None,
                "description": "",
            })
        return results
    except Exception as e:
        print(f"List images error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _rewrite_urls(row: dict) -> dict:
    url = row.get("photo_image_url", "")
    if isinstance(url, str) and url.startswith("/images/"):
        row["photo_image_url"] = f"{BACKEND_URL}{url}"
    v_url = row.get("video_url", "")
    if isinstance(v_url, str) and v_url.startswith("/images/"):
        row["video_url"] = f"{BACKEND_URL}{v_url}"
    return row


def _row_to_result(r: dict) -> dict:
    r = _rewrite_urls(dict(r))
    v_url = r.get("video_url", "")
    return {
        "photo_id": r["photo_id"],
        "photo_image_url": r["photo_image_url"],
        "video_url": v_url if v_url else None,
        "timestamp": float(r.get("timestamp", 0.0)) if pd.notna(r.get("timestamp")) else None,
        "best_timestamp": float(r.get("best_timestamp")) if r.get("best_timestamp") is not None else None,
        "description": r.get("description", ""),
        "score": float(r.get("similarity_score", 0.0)),
    }


@router.post("/search/complex", response_model=List[SearchResult])
def search_complex(req: ComplexSearchRequest):
    """Compositional search: runs existing SigLIP+OWL-ViT per clause, combines via set operations.

    mode=SINGLE  -> identical to /search on clause[0].object
    mode=OR      -> union of per-clause frame sets, score = max across clauses
    mode=AND     -> intersection of per-clause frame sets, score = min across clauses
    Negated clauses (any mode) subtract their frame set from the final result.
    """
    try:
        mode = (req.plan.mode or "SINGLE").upper()
        positive = [c for c in req.plan.clauses if not c.negated]
        negative = [c for c in req.plan.clauses if c.negated]

        if not positive:
            return []

        # Fetch wider per-clause so intersections have room. We filter by threshold at the end.
        per_clause_k = max(req.k * 2, 20)

        def _phrase(c: SearchClause) -> str:
            if c.attributes:
                return f"{' '.join(c.attributes)} {c.object}".strip()
            return c.object

        def _run_clause(c: SearchClause, is_negative: bool = False) -> list[dict]:
            phrase = _phrase(c)
            has_attrs = bool(c.attributes)
            
            # Construct pre-filter for DataFusion fast metadata search
            filters = [f"objects_json LIKE '%{c.object.lower()}%'"]
            for attr in c.attributes:
                # We split attributes to catch individual words if needed, 
                # but for Florence-2 captions, whole phrase LIKE is often better.
                filters.append(f"objects_json LIKE '%{attr.lower()}%'")
            pre_filter = " AND ".join(filters)

            # Demo-mode tuning: top-3 @ 512px for attribute clauses (half the OWL-ViT cost),
            # top-5 @ 512px otherwise. Image detection down from 768->512 is ~2x speedup.
            cache_key = f"{phrase}|attr={int(has_attrs)}"
            cached = _cache_get(cache_key)
            if cached is not None:
                return cached
            rows = search_service.search(
                phrase,
                per_clause_k,
                threshold=0.0,
                pre_filter=pre_filter,
            )
            _cache_put(cache_key, rows)
            return rows

        # Run every clause (positive + negative) in parallel. On 2-vCPU this
        # still helps: LanceDB I/O + torch GIL releases during OWL-ViT forward
        # passes give real overlap.
        pos_futures = [_CLAUSE_POOL.submit(_run_clause, c) for c in positive]
        neg_futures = [_CLAUSE_POOL.submit(_run_clause, c, True) for c in negative]
        positive_results: list[dict[str, dict]] = [
            {r["photo_id"]: r for r in f.result()} for f in pos_futures
        ]

        negative_ids: set[str] = set()
        for f in neg_futures:
            rows = f.result()
            # A frame "contains" the negated object if its clause score is meaningful.
            # Use 0.15 as the presence bar — below this the detector wasn't confident enough.
            for r in rows:
                if float(r.get("similarity_score", 0.0)) >= 0.15:
                    negative_ids.add(r["photo_id"])

        if mode == "SINGLE" or len(positive) == 1:
            combined = positive_results[0]
        elif mode == "OR":
            combined = {}
            for per in positive_results:
                for pid, row in per.items():
                    prev = combined.get(pid)
                    if prev is None or float(row.get("similarity_score", 0)) > float(prev.get("similarity_score", 0)):
                        combined[pid] = row
        elif mode == "AND":
            common_ids = set(positive_results[0].keys())
            for per in positive_results[1:]:
                common_ids &= set(per.keys())
            combined = {}
            for pid in common_ids:
                # Score = min (weakest-link); carry the row with the min score for metadata.
                rows = [per[pid] for per in positive_results]
                min_row = min(rows, key=lambda r: float(r.get("similarity_score", 0.0)))
                combined[pid] = min_row
        else:
            combined = positive_results[0]

        # Subtract negated frames.
        for pid in list(combined.keys()):
            if pid in negative_ids:
                combined.pop(pid, None)

        # Threshold + sort + cap.
        thr = float(req.threshold or 0.0)
        ordered = [
            row for row in combined.values()
            if float(row.get("similarity_score", 0.0)) >= thr
        ]
        ordered.sort(key=lambda r: float(r.get("similarity_score", 0.0)), reverse=True)
        ordered = ordered[: req.k]

        return [_row_to_result(r) for r in ordered]
    except Exception as e:
        print(f"Complex search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[SearchResult])
def search_images(req: SearchRequest):
    try:
        results = search_service.search(req.query, req.k, req.threshold)
        response = []
        for r in results:
            score = float(r.get("similarity_score", 0.0))

            url = r["photo_image_url"]
            if url.startswith("/images/"):
                url = f"{BACKEND_URL}{url}"
                
            v_url = r.get("video_url", "")
            if v_url and isinstance(v_url, str) and v_url.startswith("/images/"):
                v_url = f"{BACKEND_URL}{v_url}"

            response.append({
                "photo_id": r["photo_id"],
                "photo_image_url": url,
                "video_url": v_url if v_url else None,
                "timestamp": float(r.get("timestamp", 0.0)) if pd.notna(r.get("timestamp")) else None,
                "best_timestamp": float(r.get("best_timestamp")) if r.get("best_timestamp") is not None else None,
                "description": r.get("description", ""),
                "score": float(score)
            })
        return response
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class TrackRequest(BaseModel):
    photo_image_url: str = ""
    video_url: Optional[str] = None
    base64_image: str = None
    query: str
    video_id: Optional[str] = None

class TrackResponse(BaseModel):
    tracks: List[Dict] = [] 

@router.post("/track", response_model=TrackResponse)
def track_object(req: TrackRequest):
    try:
        import json
        
        # Try to fetch pre-computed objects from LanceDB
        if search_service.table is not None:
            url = req.photo_image_url
            if "/images/" in url:
                url_path = f"/images/{url.split('/images/')[-1]}"
                df = search_service.table.to_pandas()
                match = df[df["photo_image_url"] == url_path]
                if not match.empty:
                    objs_json = match.iloc[0].get("objects_json", "[]")
                    objs = json.loads(objs_json)
                    
                    tracks = []
                    q_words = req.query.lower().strip().split()
                    for idx, o in enumerate(objs):
                        # Combine class and attributes for full-text search
                        text_to_search = (o["class_name"] + " " + o.get("attributes", "")).lower()
                        
                        # Match only if ALL words from the query are found in the object description
                        if all(word in text_to_search for word in q_words):
                            tracks.append({
                                "track_id": idx, # Use index for unique React key
                                "bbox": o["bbox"], # [xmin, ymin, xmax, ymax] normalized
                                "score": o.get("score", 1.0)
                            })
                    if tracks:
                        print(f"DEBUG: Returning {len(tracks)} pre-computed tracks for {url_path}")
                        return TrackResponse(tracks=tracks)

        print(f"DEBUG: No pre-computed tracks found for {req.photo_image_url}")
        return TrackResponse(tracks=[])
    except Exception as e:
        print(f"Tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
