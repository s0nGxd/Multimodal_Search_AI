import os
import time
import pandas as pd
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
    timestamp: Optional[float] = None

class TrackResponse(BaseModel):
    tracks: List[Dict] = [] 

@router.post("/track", response_model=TrackResponse)
def track_object(req: TrackRequest):
    try:
        import json
        import numpy as np
        
        # 1. Determine which frame to look up
        objs = []
        source_label = ""
        
        if search_service.table is not None:
            df = search_service.table.to_pandas()
            match_df = pd.DataFrame()
            
            # Scenario A: Video Playback (lookup by video_url + closest timestamp)
            if req.video_url and req.timestamp is not None:
                v_url = req.video_url
                if "/images/" in v_url:
                    v_url = f"/images/{v_url.split('/images/')[-1]}"
                
                video_df = df[df["video_url"] == v_url]
                if not video_df.empty:
                    # Find closest timestamp within 1.5s
                    video_df = video_df.copy()
                    video_df["time_diff"] = (video_df["timestamp"] - req.timestamp).abs()
                    closest = video_df.sort_values("time_diff").iloc[0]
                    
                    if closest["time_diff"] < 1.5:
                        objs_json = closest.get("objects_json", "[]")
                        objs = json.loads(objs_json)
                        source_label = f"Video {v_url} @ {closest['timestamp']:.2f}s (diff: {closest['time_diff']:.2f}s)"
            
            # Scenario B: Static Image or Fallback (lookup by photo_image_url)
            if not objs:
                url = req.photo_image_url
                if "/images/" in url:
                    url_path = f"/images/{url.split('/images/')[-1]}"
                    match = df[df["photo_image_url"] == url_path]
                    if not match.empty:
                        objs_json = match.iloc[0].get("objects_json", "[]")
                        objs = json.loads(objs_json)
                        source_label = f"Static Image {url_path}"

        if not objs:
            if req.video_url and req.timestamp:
                print(f"DEBUG: No pre-computed frame found for {req.video_url} near {req.timestamp}s")
            return TrackResponse(tracks=[])

        # 2. Semantic Matching & Scoring
        print(f"DEBUG: Tracking Lookup [{source_label}] for query: \"{req.query}\"")
        tracks = []
        
        # Define some strict keywords to prevent over-generalization (Colors & Objects)
        STRICT_KEYWORDS = {"red", "blue", "green", "yellow", "black", "white", "grey", "gray", "brown", "person", "man", "woman", "car", "dog", "cat", "truck", "bus", "suv", "van"}
        q_words = [w for w in req.query.lower().strip().split() if w in STRICT_KEYWORDS]

        for idx, o in enumerate(objs):
            cls_name = o["class_name"].lower()
            attrs = o.get("attributes", "").lower()
            text_to_search = f"{cls_name} {attrs}"
            
            # 1. Semantic Similarity Check
            sem_score = search_service.semantic_match(req.query, attrs)
            
            # 2. Strict Keyword Sanity Check (Prevention of color-swapping or class-swapping)
            # If the user asked for "red", the word "red" must be in the text.
            # If the user asked for "car", the word "car" (or synonym) must be in the text.
            passes_strict = True
            for word in q_words:
                if word not in text_to_search:
                    passes_strict = False
                    break
            
            # Threshold 0.55 + Strict check ensures high-quality matches
            if sem_score > 0.55 and passes_strict:
                print(f"    - Match! Object {idx}: [{o['class_name']}] Semantic Score: {sem_score:.2f} | Text: {attrs[:60]}...")
                
                tracks.append({
                    "track_id": idx,
                    "bbox": o["bbox"],
                    "score": float(o.get("score", 1.0)) * sem_score
                })

        # Sort by relevance and return
        if tracks:
            tracks.sort(key=lambda x: x["score"], reverse=True)
            print(f"DEBUG: Returning {len(tracks)} semantic matches")
            return TrackResponse(tracks=tracks)

        return TrackResponse(tracks=[])
    except Exception as e:
        print(f"Tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
