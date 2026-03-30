from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from app.services.persistence_service import restore_from_repo

@asynccontextmanager
async def lifespan(app):
    """Startup: restore persisted data from HF Dataset repo."""
    restore_from_repo()
    yield

app = FastAPI(title="SEGP Semantic Search API", lifespan=lifespan)

# CORS Setup — configurable for deployment
# Locally: defaults to localhost:3000
# Deployed: set ALLOWED_ORIGINS env var to your frontend URL (comma-separated)
default_origins = "http://localhost:3000,http://127.0.0.1:3000,https://client-nine-xi-31.vercel.app"
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
def health_check():
    return {"status": "ok", "service": "semantic-search-api"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

