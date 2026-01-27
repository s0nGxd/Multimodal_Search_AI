import os
import shutil
from pathlib import Path
from PIL import Image
import lancedb
from lancedb.pydantic import LanceModel, Vector
from .search_service import search_service
from .caption_service import caption_service

# Define Schema (Must match what is used in SearchService/Ingest)
class ImageRecord(LanceModel):
    photo_id: str
    photo_image_url: str
    description: str = ""
    vector: Vector(512)

import requests
import io
import pandas as pd
from typing import List, Optional

class IngestionService:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def process_upload(self, file_contents: bytes, filename: str):
        # 1. Save File
        file_path = self.data_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_contents)
        
        # 2. Open and Validate Image
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            if file_path.exists():
                os.remove(file_path)
            raise ValueError(f"Invalid image file: {e}")

        # 3. Embed and Caption Image
        vector = search_service.embed_image(img)
        description = caption_service.generate_caption(img)
        
        # 4. Insert into LanceDB
        photo_url = f"http://localhost:8000/images/{filename}"
        record = ImageRecord(
            photo_id=file_path.stem,
            photo_image_url=photo_url,
            description=description,
            vector=vector
        )
        self._insert_records([record])
        
        return {"id": file_path.stem, "url": photo_url, "description": description, "status": "indexed"}

    def process_url_upload(self, url: str, photo_id: Optional[str] = None):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to fetch image from URL: {e}")

        vector = search_service.embed_image(img)
        description = caption_service.generate_caption(img)
        pid = photo_id or f"remote_{hash(url)}"
        
        record = ImageRecord(
            photo_id=pid,
            photo_image_url=url,
            description=description,
            vector=vector
        )
        self._insert_records([record])
        return {"id": pid, "url": url, "description": description, "status": "indexed"}

    def process_bulk_csv(self, csv_path: str, limit: int = 100):
        # Load CSV
        try:
            # The file is tab separated based on head output
            df = pd.read_csv(csv_path, sep='\t', nrows=limit)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        records = []
        for _, row in df.iterrows():
            photo_id = str(row['photo_id'])
            image_url = row['photo_image_url']
            
            try:
                # We don't download everything here for speed, 
                # but if we want vectors, we HAVE to download and embed.
                # In a real system, this would be a background task.
                response = requests.get(image_url, timeout=5)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                vector = search_service.embed_image(img)
                
                records.append(ImageRecord(
                    photo_id=photo_id,
                    photo_image_url=image_url,
                    vector=vector
                ))
            except Exception as e:
                print(f"Skipping {photo_id} due to error: {e}")
                continue
        
        if records:
            self._insert_records(records)
        
        return {"processed": len(records), "status": "completed"}

    def _insert_records(self, records: List[ImageRecord]):
        db = lancedb.connect(search_service.db_uri)
        try:
            table = db.open_table("images")
            table.add(records)
        except:
            db.create_table("images", schema=ImageRecord, data=records)
        search_service.refresh_table()

ingestion_service = IngestionService()
