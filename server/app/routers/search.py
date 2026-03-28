import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.search_service import search_service

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

router = APIRouter()

# Hybrid search (SigLIP vectors + BM25) produces cosine similarities from roughly 0.15 to 1.0.
# We rescale this range to 0-100% for human-readable display.
# SIM_FLOOR was lowered from 0.40 (CLIP) to 0.15 to correctly spread SigLIP + BM25 scores.
SIM_FLOOR = 0.15  # Below this = 0% (unrelated noise)
SIM_CEIL = 1.00   # Perfect match = 100%


def rescale_score(raw_cosine_distance: float) -> float:
    raw_sim = 1.0 - raw_cosine_distance
    scaled = (raw_sim - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR)
    return max(0.0, min(1.0, scaled))


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 20
    threshold: Optional[float] = 0.75  # Cosine distance threshold

class SearchResult(BaseModel):
    photo_id: str
    photo_image_url: str
    description: Optional[str] = None
    score: float

@router.get("/images/all")
async def list_all_images():
    try:
        if search_service.table is None:
            search_service.refresh_table()
            if search_service.table is None:
                return []
        df = search_service.table.to_pandas()
        results = []
        for _, row in df.iterrows():
            url = row["photo_image_url"]
            if url.startswith("/images/"):
                url = f"{BACKEND_URL}{url}"
            results.append({
                "photo_id": row["photo_id"],
                "photo_image_url": url,
                "description": row.get("description", ""),
            })
        return results
    except Exception as e:
        print(f"List images error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=List[SearchResult])
async def search_images(req: SearchRequest):
    try:
        results = search_service.search(req.query, req.k, req.threshold)
        response = []
        for r in results:
            # Use normalised RRF score: reflects how highly this image ranked across
            # ALL 3 search channels (image vector + caption vector + BM25 keyword).
            # Top result = 1.0, others proportionally lower. Far more intuitive than
            # raw cosine distance, and directly rewards keyword/object matches.
            score = float(r.get("_rrf_score", 0.0))

            url = r["photo_image_url"]
            if url.startswith("/images/"):
                url = f"{BACKEND_URL}{url}"

            response.append({
                "photo_id": r["photo_id"],
                "photo_image_url": url,
                "description": r.get("description", ""),
                "score": float(score)
            })
        return response
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
