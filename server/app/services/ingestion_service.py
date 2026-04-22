import os
import io
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests
import pandas as pd
from PIL import Image
import lancedb
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv

from .search_service import search_service
from .caption_service import caption_service
from .detection_service import detection_service
from .persistence_service import sync_to_repo

load_dotenv()

# SigLIP Base uses 768 dimension
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

class ImageRecord(LanceModel):
    photo_id: str
    photo_image_url: str
    video_url: str = ""
    timestamp: float = 0.0
    description: str = ""
    vector: Vector(EMBED_DIM)
    caption_vector: Vector(EMBED_DIM)
    
    # ── Dense Region Fields ──
    is_region: bool = False
    parent_photo_id: str = ""
    bbox: str = ""          # Stored as JSON string "[x1, y1, x2, y2]"
    region_label: str = ""  # The dense region caption for this specific crop


def _download_image(photo_id: str, url: str, timeout: int = 10) -> tuple[str, str, Image.Image | None]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return (photo_id, url, img)
    except Exception as e:
        print(f"Skipping {photo_id}: {e}")
        return (photo_id, url, None)


class IngestionService:
    def __init__(self):
        self.data_dir = Path(os.getenv("DATA_DIR", "data"))
        self.data_dir.mkdir(exist_ok=True)

    def _extract_regions(self, img: Image.Image, original_id: str, photo_url: str, video_url: str = "", timestamp: float = 0.0) -> List[ImageRecord]:
        import json
        region_records = []
        
        # 1. Use YOLO-World Large + SAHI to find all dense objects.
        # Fixed SAHI configuration handles large objects (like the black dog) correctly.
        detected_objects = detection_service.detect_dense(img)
        
        for i, obj in enumerate(detected_objects):
            try:
                box = obj["bbox"] # [x1, y1, x2, y2]
                base_label = obj["label"]
                
                # 2. Use Florence-2 to describe the region in the context of the FULL image.
                # This ensures attributes like color and action are preserved (e.g. "yellow car").
                detailed_caption = caption_service.region_to_description(img, box)
                
                if not detailed_caption:
                    detailed_caption = base_label
                
                # Prepend the YOLO base label so the canonical name is always searchable.
                if base_label.lower() not in detailed_caption.lower():
                    detailed_caption = f"{base_label}. {detailed_caption}"
                    
                # 3. Dual-Encoder: Generate visual vector (focused on the object) 
                # and text attribute vector (context-aware description).
                crop_img = img.crop(box)
                # Ignore tiny crops that will crash models
                if crop_img.width < 10 or crop_img.height < 10:
                    continue

                vector = search_service.embed_image(crop_img)
                caption_vector = search_service.embed_text(detailed_caption)
                
                # Normalize box coordinates to 0..1 range for the frontend tracking
                w, h = img.size
                norm_box = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]
                
                region_records.append(ImageRecord(
                    photo_id=f"{original_id}_region_{i}",
                    photo_image_url=photo_url,
                    video_url=video_url,
                    timestamp=timestamp,
                    description=detailed_caption,
                    vector=vector,
                    caption_vector=caption_vector,
                    is_region=True,
                    parent_photo_id=original_id,
                    bbox=json.dumps(norm_box),
                    region_label=detailed_caption
                ))
            except Exception as e:
                print(f"Failed to process region {i} for {original_id}: {e}")
        return region_records

    def process_upload(self, file_contents: bytes, filename: str):
        original_id = Path(filename).stem
        if search_service.table is not None:
            df = search_service.table.to_pandas()
            exists = any(df['photo_id'] == original_id) or any(df['video_url'].str.contains(filename, na=False))
            if exists:
                print(f"Skipping {filename}: already exists in database.")
                return {"id": original_id, "status": "skipped", "message": "Already exists"}

        timestamp = int(pd.Timestamp.now().timestamp())
        path_obj = Path(filename)
        unique_filename = f"{path_obj.stem}_{timestamp}{path_obj.suffix}"
        
        file_path = self.data_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(file_contents)

        mime_type, _ = mimetypes.guess_type(unique_filename)
        ext = path_obj.suffix.lower()
        is_video = (mime_type and mime_type.startswith("video/")) or ext in [".mp4", ".mov", ".avi", ".webm", ".mkv"]

        if is_video:
            return self._process_video_upload(file_path, unique_filename)

        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            if file_path.exists():
                os.remove(file_path)
            raise ValueError(f"Invalid image file: {e}")

        # ── Global Frame ──
        vector = search_service.embed_image(img)
        description = caption_service.generate_caption(img)
        caption_vector = search_service.embed_text(description)

        photo_url = f"/images/{unique_filename}"
        global_record = ImageRecord(
            photo_id=original_id,
            photo_image_url=photo_url,
            description=description,
            vector=vector,
            caption_vector=caption_vector,
        )
        
        # ── Dense Regions ──
        records = [global_record]
        region_records = self._extract_regions(img, original_id, photo_url)
        records.extend(region_records)

        self._insert_records(records)
        sync_to_repo()
        return {"id": original_id, "url": photo_url, "description": f"Image indexed with {len(region_records)} regions", "status": "indexed"}

    def _process_video_upload(self, file_path: Path, filename: str):
        import cv2
        cap = cv2.VideoCapture(str(file_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        frames_to_extract = []
        frame_interval = int(fps * 2) # 1 frame every 2 seconds
        
        count = 0
        success, frame = cap.read()
        while success:
            if count % frame_interval == 0:
                frames_to_extract.append((count, frame))
            success, frame = cap.read()
            count += 1
        cap.release()
        
        if not frames_to_extract:
            if file_path.exists():
                os.remove(file_path)
            raise ValueError("Could not extract any frames from the video")
            
        records = []
        for frame_idx, frame_data in frames_to_extract:
            frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            frame_filename = f"{file_path.stem}_frame_{frame_idx}.jpg"
            frame_path = self.data_dir / frame_filename
            img.save(frame_path)
            
            # ── Global Frame ──
            vector = search_service.embed_image(img)
            description = caption_service.generate_caption(img)
            caption_vector = search_service.embed_text(description)
            
            photo_url = f"/images/{frame_filename}"
            global_record = ImageRecord(
                photo_id=f"{file_path.stem}_frame_{frame_idx}",
                photo_image_url=photo_url,
                video_url=f"/images/{filename}",
                timestamp=float(frame_idx) / float(fps),
                description=description,
                vector=vector,
                caption_vector=caption_vector,
            )
            records.append(global_record)
            
            # ── Dense Regions ──
            region_records = self._extract_regions(
                img, 
                original_id=f"{file_path.stem}_frame_{frame_idx}",
                photo_url=photo_url,
                video_url=f"/images/{filename}",
                timestamp=float(frame_idx) / float(fps)
            )
            records.extend(region_records)
            
        if records:
            self._insert_records(records)
            sync_to_repo()
            
        return {
            "id": file_path.stem, 
            "url": f"/images/{filename}", 
            "description": f"Video indexed with {len(records)} records (frames + regions)", 
            "status": "indexed"
        }

    def process_url_upload(self, url: str, photo_id: Optional[str] = None):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to fetch image from URL: {e}")

        vector = search_service.embed_image(img)
        description = caption_service.generate_caption(img)
        caption_vector = search_service.embed_text(description)
        pid = photo_id or f"remote_{hash(url)}"

        record = ImageRecord(
            photo_id=pid,
            photo_image_url=url,
            description=description,
            vector=vector,
            caption_vector=caption_vector,
        )
        self._insert_records([record])
        sync_to_repo()
        return {"id": pid, "url": url, "description": "URL image indexed", "status": "indexed"}

    def process_bulk_csv(self, csv_path: str, limit: int = 100, generate_captions: bool = False, max_workers: int = 8, batch_size: int = 16):
        try:
            df = pd.read_csv(csv_path, sep='\t', nrows=limit)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        print(f"Downloading up to {len(df)} images with {max_workers} threads...")
        downloaded = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_image, str(row['photo_id']), row['photo_image_url']): i
                for i, row in df.iterrows()
            }
            for future in as_completed(futures):
                photo_id, url, img = future.result()
                if img is not None:
                    downloaded.append((photo_id, url, img))

        if not downloaded:
            return {"processed": 0, "status": "completed"}

        print(f"Downloaded {len(downloaded)} images. Generating embeddings in batches of {batch_size}...")
        images = [img for _, _, img in downloaded]
        vectors = search_service.embed_images_batch(images, batch_size=batch_size)

        import numpy as np
        zero_vec = np.zeros(EMBED_DIM, dtype="float32")
        records = []
        for i, (photo_id, url, img) in enumerate(downloaded):
            description = ""
            cap_vec = zero_vec
            if generate_captions:
                description = caption_service.generate_caption(img)
                cap_vec = search_service.embed_text(description)

            records.append(ImageRecord(
                photo_id=photo_id,
                photo_image_url=url,
                description=description,
                vector=vectors[i],
                caption_vector=cap_vec,
            ))

        print(f"Inserting {len(records)} records into LanceDB...")
        self._insert_records(records)
        self._maybe_create_index()
        sync_to_repo()
        return {"processed": len(records), "status": "completed"}

    def _insert_records(self, records: List[ImageRecord]):
        db = lancedb.connect(search_service.db_uri)
        try:
            table = db.open_table("images")
            table.add(records)
        except Exception:
            table = db.create_table("images", schema=ImageRecord, data=records)

        try:
            table.create_fts_index("description", replace=True)
        except Exception as e:
            print(f"FTS index creation skipped: {e}")

        search_service.refresh_table()

    def _maybe_create_index(self, min_rows: int = 256):
        try:
            table = search_service.table
            if table is None:
                return
            row_count = table.count_rows()
            if row_count < min_rows:
                print(f"Skipping index creation: {row_count} rows < {min_rows} minimum")
                return
            num_partitions = max(2, int(row_count ** 0.5))
            num_sub_vectors = min(16, EMBED_DIM) 
            print(f"Building IVF-PQ index: {row_count} rows...")
            table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                replace=True,
            )
            print("IVF-PQ index built successfully.")
        except Exception as e:
            print(f"Index creation skipped or failed: {e}")

ingestion_service = IngestionService()
