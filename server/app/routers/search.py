import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.search_service import search_service

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 20
    threshold: Optional[float] = 0.20  # Min Similarity (0.0 - 1.0)

class SearchResult(BaseModel):
    photo_id: str
    photo_image_url: str
    video_url: Optional[str] = None
    timestamp: Optional[float] = None
    best_timestamp: Optional[float] = None
    description: Optional[str] = None
    score: float

class DetectRequest(BaseModel):
    photo_image_url: str = ""
    base64_image: str = None
    query: str

class DetectResponse(BaseModel):
    box: Optional[List[float]] = None
    score: Optional[float] = None

class VideoFrameInfo(BaseModel):
    timestamp: float
    description: str

@router.get("/video/frames", response_model=List[VideoFrameInfo])
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
        has_video = df["video_url"].astype(str).str.len().gt(0)
        videos = df[has_video].drop_duplicates(subset=["video_url"], keep="first")
        photos = df[~has_video]
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
        from app.services.tracking_service import tracking_service
        import io
        import os
        import base64
        from PIL import Image
        
        if req.base64_image:
            image_data = base64.b64decode(req.base64_image.split(",")[1] if "," in req.base64_image else req.base64_image)
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            url = req.photo_image_url or (req.video_url or "")
            if "/images/" in url:
                file_name = url.split("/images/")[-1]
                local_path = os.path.join("data", file_name)
                if os.path.exists(local_path):
                    img = Image.open(local_path).convert("RGB")
                else:
                    raise FileNotFoundError(f"Local image not found: {local_path}")
            else:
                import requests
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
        
        video_id = req.video_id or req.video_url or req.photo_image_url
        tracking_service.reset_for_new_video(video_id, req.query)
        
        tracks = tracking_service.detect_and_track(img, req.query, return_all_detections=True)
        
        return TrackResponse(tracks=tracks)
    except Exception as e:
        print(f"Tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect", response_model=DetectResponse)
def detect_object(req: DetectRequest):
    try:
        from app.services.detection_service import detection_service
        import io
        import os
        import base64
        from PIL import Image
        
        if req.base64_image:
            image_data = base64.b64decode(req.base64_image.split(",")[1] if "," in req.base64_image else req.base64_image)
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            url = req.photo_image_url
            if "/images/" in url:
                file_name = url.split("/images/")[-1]
                local_path = os.path.join("data", file_name)
                if os.path.exists(local_path):
                    img = Image.open(local_path).convert("RGB")
                else:
                    raise FileNotFoundError(f"Local image not found: {local_path}")
            else:
                import requests
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
        
        # USE FULL QUALITY for image detection
        result = detection_service.detect(img, req.query)
        if result:
            return DetectResponse(box=result["box"], score=result["score"])
        return DetectResponse(box=None, score=None)
    except Exception as e:
        print(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
