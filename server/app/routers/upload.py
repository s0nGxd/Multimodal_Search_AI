import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingestion_service import ingestion_service
from app.services.search_service import search_service

from pydantic import BaseModel
from typing import Optional

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

router = APIRouter()

class URLIngestRequest(BaseModel):
    url: str
    photo_id: Optional[str] = None

class BulkIngestRequest(BaseModel):
    csv_path: str = "../photos.csv000"
    limit: int = 50
    generate_captions: bool = False

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = ingestion_service.process_upload(contents, file.filename)
        if result.get("url", "").startswith("/images/"):
            result["url"] = f"{BACKEND_URL}{result['url']}"
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/url")
async def ingest_url(req: URLIngestRequest):
    try:
        result = ingestion_service.process_url_upload(req.url, req.photo_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear-db")
async def clear_database():
    try:
        db_path = search_service.db_uri
        if db_path and shutil.os.path.exists(db_path):
            shutil.rmtree(db_path)
        search_service.db = __import__("lancedb").connect(db_path)
        search_service.table = None
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/bulk")
async def ingest_bulk(req: BulkIngestRequest):
    try:
        # We run this in the background ideally, but for now simple call
        result = ingestion_service.process_bulk_csv(req.csv_path, req.limit, generate_captions=req.generate_captions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
