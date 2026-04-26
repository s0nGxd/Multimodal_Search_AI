# Load .env before any service module imports, so os.getenv() at module-load
# time in persistence_service / search_service / etc. sees the values.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from app.services.persistence_service import restore_from_repo

@asynccontextmanager
async def lifespan(app):
    """Startup: restore persisted data + warm up models."""
    restore_from_repo()
    try:
        from app.services.search_service import search_service
        from app.services.detection_service import detection_service
        from PIL import Image
        # Dummy forward pass on both models so first real query is fast
        _ = search_service.embed_text("warmup")
        dummy = Image.new("RGB", (64, 64), color=(0, 0, 0))
        _ = detection_service.detect(dummy, "warmup")
        print("Models warmed up (SigLIP + OWL-ViT).")
    except Exception as e:
        print(f"Model warmup skipped/failed: {e}")
    yield

app = FastAPI(title="SEGP Semantic Search API", lifespan=lifespan)

# CORS Setup — configurable for deployment
# Locally: defaults to localhost:3000
# Deployed: set ALLOWED_ORIGINS env var to your frontend URL (comma-separated)
default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,https://client-nine-xi-31.vercel.app"
origins = os.getenv("ALLOWED_ORIGINS", default_origins).split(",")
# Include any Vercel preview/production URLs
origins = [o.strip() for o in origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists (uses /data for persistent storage on HF Spaces)
data_dir = os.getenv("DATA_DIR", "data")
os.makedirs(data_dir, exist_ok=True)

class CORSStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Access-Control-Allow-Origin"] = "*"
        # Aggressive cache — filenames include timestamps so content is immutable per URL.
        response.headers["Cache-Control"] = "public, max-age=86400, immutable"
        return response

# Mount static files (uploaded images) with CORS injection
app.mount("/images", CORSStaticFiles(directory=data_dir), name="images")

from app.routers import search, upload

app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])

@app.get("/")
def root():
    return {"message": "SEGP Semantic Search API", "docs": "/docs", "health": "/health"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "semantic-search-api"}

@app.get("/ready")
async def readiness_check():
    try:
        from app.services.search_service import search_service
        from app.services.detection_service import detection_service
        from PIL import Image
        _ = search_service.embed_text("ready")
        dummy = Image.new("RGB", (64, 64), color=(0, 0, 0))
        _ = detection_service.detect(dummy, "ready")
        return {"status": "ready", "service": "semantic-search-api"}
    except Exception as e:
        from fastapi import Response
        import json
        return Response(
            content=json.dumps({"status": "warming", "detail": str(e)[:200]}),
            status_code=503,
            media_type="application/json",
        )

@app.delete("/api/images/{photo_id}")
async def delete_image(photo_id: str):
    try:
        from app.services.search_service import search_service
        if search_service.table is not None:
            # First, check if this is part of a video
            df = search_service.table.to_pandas()
            target = df[df["photo_id"] == photo_id]
            
            if target.empty:
                return {"status": "not_found", "photo_id": photo_id}
            
            video_url = target.iloc[0].get("video_url", "")
            
            if video_url and isinstance(video_url, str) and video_url.strip():
                # Delete all frames associated with this video
                search_service.table.delete(f"video_url = '{video_url}'")
                return {"status": "deleted_video", "video_url": video_url}
            else:
                # Regular image delete
                search_service.table.delete(f"photo_id = '{photo_id}'")
                return {"status": "deleted", "photo_id": photo_id}
        else:
            raise HTTPException(status_code=500, detail="Database table not initialized")
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

