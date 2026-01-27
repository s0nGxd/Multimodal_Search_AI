from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingestion_service import ingestion_service

from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class URLIngestRequest(BaseModel):
    url: str
    photo_id: Optional[str] = None

class BulkIngestRequest(BaseModel):
    csv_path: str = "../photos.csv000"
    limit: int = 50

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = ingestion_service.process_upload(contents, file.filename)
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

@router.post("/ingest/bulk")
async def ingest_bulk(req: BulkIngestRequest):
    try:
        # We run this in the background ideally, but for now simple call
        result = ingestion_service.process_bulk_csv(req.csv_path, req.limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
