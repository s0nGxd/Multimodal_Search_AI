import os
import io
import json
import gc
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

import requests
import pandas as pd
import numpy as np
from PIL import Image
import lancedb
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv

from .search_service import search_service
from .persistence_service import sync_to_repo

load_dotenv()

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Must match the embedding dimension of the configured model.
# SigLIP base-patch16-256 (default) → 768. CLIP base-patch32 → 512.
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))


class ImageRecord(LanceModel):
    photo_id: str
    photo_image_url: str
    video_url: str = ""
    timestamp: float = 0.0
    vector: Vector(EMBED_DIM)
    objects_json: str = "[]"


_DL_HEADERS = {
    "User-Agent": "SEMANTIX-Ingest/1.0 (academic; aarizsajan2@gmail.com)",
    "Accept": "image/*,*/*;q=0.8",
}


def _download_image(photo_id: str, url: str, timeout: int = 15) -> tuple[str, str, Image.Image | None]:
    try:
        response = requests.get(url, timeout=timeout, headers=_DL_HEADERS)
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

    def process_upload(self, file_contents: bytes, filename: str):
        # 1. Check if this file (by original stem) already exists in DB
        # This prevents re-indexing the same file multiple times
        original_id = Path(filename).stem
        try:
            if search_service.table is not None:
                df = search_service.table.to_pandas()
                # Match stored photo_id patterns:
                #   still image: "<stem>_<timestamp>"
                #   video frame: "<stem>_frame_<n>"
                # Also match stored video_url which contains the original stem as a substring.
                exists = (
                    any(df['photo_id'] == original_id)
                    or any(df['photo_id'].str.startswith(f"{original_id}_", na=False))
                    or any(df['video_url'].str.contains(f"/{original_id}_", na=False, regex=False))
                )
                if exists:
                    print(f"Skipping {filename}: already exists in database.")
                    return {"id": original_id, "status": "skipped", "message": "Already exists"}
        except Exception as e:
            print(f"Warning: Could not check for duplicates (possible DB corruption): {e}")
            search_service.refresh_table()

        # 2. Generate unique filename for storage safety
        timestamp = int(pd.Timestamp.now().timestamp())
        path_obj = Path(filename)
        unique_filename = f"{path_obj.stem}_{timestamp}{path_obj.suffix}"
        
        file_path = self.data_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(file_contents)

        import mimetypes
        mime_type, _ = mimetypes.guess_type(unique_filename)
        ext = path_obj.suffix.lower()
        is_video = (mime_type and mime_type.startswith("video/")) or ext in [".mp4", ".mov", ".avi", ".webm", ".mkv"]

        if is_video:
            return self._run_ingestion_pipeline(file_path, unique_filename, is_video=True)
        else:
            return self._run_ingestion_pipeline(file_path, unique_filename, is_video=False)

    def _run_ingestion_pipeline(self, file_path: Path, filename: str, is_video: bool):
        import cv2
        print(f"DEBUG: Processing upload: {filename}")
        frames = [] # List of (timestamp, Image)

        if is_video:
            # --- STAGE 0: Frame Extraction (1 FPS) ---
            print("DEBUG: Stage 0 - Extracting frames at 1 FPS")
            cap = cv2.VideoCapture(str(file_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            count = 0
            success, frame = cap.read()
            while success:
                if count % int(fps) == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((float(count) / fps, Image.fromarray(frame_rgb)))
                success, frame = cap.read()
                count += 1
            cap.release()
        else:
            # --- STAGE 0: Image Loading ---
            print("DEBUG: Stage 0 - Loading image")
            try:
                img = Image.open(file_path).convert("RGB")
                frames.append((0.0, img))
            except Exception as e:
                if file_path.exists():
                    os.remove(file_path)
                raise ValueError(f"Invalid image file: {e}")
        
        if not frames:
            if file_path.exists():
                os.remove(file_path)
            raise ValueError("Could not extract any data from the file")
            
        # --- STAGE 1: SigLIP Scene Embedding ---
        print(f"DEBUG: Stage 1 - Generating scene embeddings with SigLIP-2 for {len(frames)} frames")
        search_service.load_model()
        vectors = []
        try:
            for _, img in frames:
                vectors.append(search_service.embed_image(img))
        finally:
            search_service.unload_model()
            clear_vram()

        # --- STAGE 2: YOLO + SAHI Detection & Tracking ---
        print("DEBUG: Stage 2 - Detecting objects with YOLOv8 + SAHI")
        from ultralytics import YOLO
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        
        yolo_model = YOLO("yolov8n.pt")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model=yolo_model,
            confidence_threshold=0.25,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        all_frame_objects = []
        try:
            for ts, img in frames:
                result = get_sliced_prediction(
                    img,
                    detection_model,
                    slice_height=512,
                    slice_width=512,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    verbose=0
                )
                objs = []
                for pred in result.object_prediction_list:
                    # Filter for relevant classes
                    cls = pred.category.name.lower()
                    if cls in ["person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat", "horse", "sheep", "cow"]:
                        objs.append({
                            "class_name": cls,
                            "bbox": pred.bbox.to_xyxy(),
                            "score": float(pred.score.value),
                            "track_id": 0 # Simple ID for now
                        })
                all_frame_objects.append(objs)
        finally:
            del detection_model
            del yolo_model
            clear_vram()

        # --- STAGE 3: Florence-2 Attribute Extraction ---
        print("DEBUG: Stage 3 - Extracting attributes with Florence-2")
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        f2_model_id = "microsoft/Florence-2-base"
        f2_processor = AutoProcessor.from_pretrained(f2_model_id, trust_remote_code=True)
        f2_model = AutoModelForCausalLM.from_pretrained(f2_model_id, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        
        records = []
        try:
            for i, (ts, img) in enumerate(frames):
                frame_objs = all_frame_objects[i]
                for obj in frame_objs:
                    x1, y1, x2, y2 = obj["bbox"]
                    # Ensure bbox is within image bounds
                    w, h = img.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1: continue

                    # Normalize bbox for frontend [xmin, ymin, xmax, ymax]
                    obj["bbox"] = [float(x1/w), float(y1/h), float(x2/w), float(y2/h)]
                    
                    crop = img.crop((x1, y1, x2, y2))
                    
                    # Florence-2 task-based captioning
                    # The processor requires the task token to be the ONLY token in the text.
                    if obj["class_name"] == "person":
                        prompt = "<MORE_DETAILED_CAPTION>"
                    elif obj["class_name"] in ["car", "bicycle", "motorcycle", "bus", "truck"]:
                        prompt = "<DETAILED_CAPTION>"
                    else: # Animals
                        prompt = "<DETAILED_CAPTION>"
                    
                    inputs = f2_processor(text=prompt, images=crop, return_tensors="pt").to(f2_model.device)
                    generated_ids = f2_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=256,
                        num_beams=3
                    )
                    response = f2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    obj["attributes"] = response
                    print(f"    [Frame {i}] Detected {obj['class_name']} (conf: {obj['score']:.2f}) -> {response}")
                
                # Save frame image
                if is_video:
                    frame_filename = f"{file_path.stem}_frame_{i}.jpg"
                    frame_path = self.data_dir / frame_filename
                    img.save(frame_path)
                else:
                    frame_filename = filename
                
                records.append(ImageRecord(
                    photo_id=f"{file_path.stem}_frame_{i}" if is_video else file_path.stem,
                    photo_image_url=f"/images/{frame_filename}",
                    video_url=f"/images/{filename}" if is_video else "",
                    timestamp=ts,
                    vector=vectors[i],
                    objects_json=json.dumps(frame_objs)
                ))
        finally:
            del f2_model
            del f2_processor
            clear_vram()
            
        if records:
            print(f"DEBUG: Inserting {len(records)} frames into database")
            self._insert_records(records)
            sync_to_repo()
            
        return {
            "id": file_path.stem,
            "url": f"/images/{filename}",
            "status": "indexed",
            "frames": len(records),
        }

    def process_url_upload(self, url: str, photo_id: Optional[str] = None):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to fetch image from URL: {e}")

        # Generate unique filename
        timestamp = int(pd.Timestamp.now().timestamp())
        filename = f"url_{timestamp}.jpg"
        file_path = self.data_dir / filename
        img.save(file_path)

        return self._run_ingestion_pipeline(file_path, filename, is_video=False)

    def process_bulk_csv(self, csv_path: str, limit: int = 100, max_workers: int = 3, batch_size: int = 16, offset: int = 0):
        try:
            df = pd.read_csv(csv_path, sep='\t', skiprows=range(1, offset + 1) if offset else None, nrows=limit)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        existing_ids: set[str] = set()
        if search_service.table is not None:
            try:
                existing_ids = set(search_service.table.to_pandas()['photo_id'].astype(str).tolist())
            except Exception:
                existing_ids = set()
        df = df[~df['photo_id'].astype(str).isin(existing_ids)].reset_index(drop=True)
        if df.empty:
            return {"processed": 0, "skipped_existing": True, "status": "completed"}

        # Phase 1: Download images concurrently (low concurrency to be polite to upstream)
        print(f"Downloading {len(df)} images with {max_workers} threads (offset={offset})...")
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

        # Phase 2: Batch embed all images through SigLIP
        images = [img for _, _, img in downloaded]
        vectors = search_service.embed_images_batch(images, batch_size=batch_size)

        records = [
            ImageRecord(photo_id=photo_id, photo_image_url=url, vector=vectors[i])
            for i, (photo_id, url, _) in enumerate(downloaded)
        ]

        # Phase 3: Insert all records at once
        print(f"Inserting {len(records)} records into LanceDB...")
        self._insert_records(records)

        # Phase 4: Build IVF-PQ index if table is large enough
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
            num_sub_vectors = 16
            print(f"Building IVF-PQ index: {row_count} rows, {num_partitions} partitions, {num_sub_vectors} sub-vectors...")
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
