from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="SEGP Semantic Search API")

# CORS Setup
origins = [
    "http://localhost:3000",  # Next.js frontend
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Mount static files (uploaded images)
# We will serve files from the 'data' directory at the /images URL path
app.mount("/images", StaticFiles(directory="data"), name="images")

from app.routers import search, upload

app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "semantic-search-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
