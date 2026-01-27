from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.search_service import search_service

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 20
    threshold: Optional[float] = 0.5  # Cosine distance threshold

class SearchResult(BaseModel):
    photo_id: str
    photo_image_url: str
    score: float

@router.post("/search", response_model=List[SearchResult])
async def search_images(req: SearchRequest):
    try:
        results = search_service.search(req.query, req.k, req.threshold)
        response = []
        for r in results:
            # Score here is 1 - distance (to make it a similarity score for the UI)
            # Or we can just return distance. 1 - distance is usually more intuitive (1.0 = perfect match)
            dist = r.get("_distance", 1.0)
            similarity = max(0, 1 - dist)
            
            response.append({
                "photo_id": r["photo_id"],
                "photo_image_url": r["photo_image_url"],
                "score": float(similarity)
            })
        return response
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
