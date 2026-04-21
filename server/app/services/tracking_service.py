import os
import re
import shutil
import tempfile
import time
import torch
import cv2
import numpy as np
from typing import List, Dict, Optional
from PIL import Image

try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 is not installed. Falling back to frame-by-frame Grounding DINO.")

# ── Filler words to strip before sending queries to GDINO ─────────────────
_FILLER = {
    'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'for',
    'is', 'to', 'by', 'from', 'its', 'this', 'that',
    'color', 'colour', 'colored', 'coloured', 'looking', 'like', 'type',
    'kind', 'style', 'very', 'really', 'quite', 'pretty', 'beautiful',
}


def _clean_query_for_gdino(query: str) -> str:
    """Strip filler words that confuse GDINO.

    'brown color dog running' → 'brown dog running'
    'a photo of a large dog'  → 'large dog'
    """
    words = query.strip().split()
    cleaned = [w for w in words if w.lower().strip(".,!?") not in _FILLER]
    return " ".join(cleaned) if cleaned else query.strip()


class TrackingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrackingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        from app.services.detection_service import detection_service
        self.detection_service = detection_service

        self.predictor = None
        self.sam2_checkpoint = os.getenv("SAM2_CHECKPOINT", None)
        if not self.sam2_checkpoint and SAM2_AVAILABLE:
            self.sam2_cfg = "sam2_hiera_s.yaml"

    # ── Single-frame detection (for images + legacy fallback) ────────────

    def detect_and_track(self, frame: Image.Image, query: str,
                         return_all_detections: bool = False) -> List[Dict]:
        """Single-frame detection via Grounding DINO.  verify=False for speed."""
        clean_q = _clean_query_for_gdino(query)

        if return_all_detections:
            detections = self.detection_service.detect_multiple(
                frame, clean_q, threshold=0.25, verify=False
            )
            return [
                {"track_id": i + 1, "bbox": d["box"], "score": d["score"]}
                for i, d in enumerate(detections)
            ]
        result = self.detection_service.detect(frame, clean_q, verify=False)
        if result and result["score"] >= 0.15:
            return [{"track_id": 1, "bbox": result["box"], "score": result["score"]}]
        return []

    def reset_for_new_video(self, video_id: str, query: str):
        """No-op — kept for API compatibility."""
        pass

    # ── Video preloading (the core new feature) ──────────────────────────

    def preload_video_tracks(self, video_path: str, query: str,
                             fps_sample: float = 1.0,
                             max_width: int = 640) -> Dict:
        """Extract keyframes from a video and run GDINO on each one.

        Returns a timeline of bounding boxes that the frontend can interpolate
        between for smooth, lag-free tracking during playback.

        Parameters
        ----------
        video_path : str
            Local filesystem path to the video file.
        query : str
            The user's search query (e.g. "brown dog").
        fps_sample : float
            How many keyframes to extract per second of video. Default 1.0.
        max_width : int
            Resize frames to this width before running GDINO (speed optimisation).

        Returns
        -------
        dict with:
            keyframes: list of {time: float, tracks: [{track_id, bbox, score}]}
            duration: float (total video length in seconds)
        """
        clean_q = _clean_query_for_gdino(query)
        t0 = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        # Calculate which frame indices to sample
        frame_interval = max(1, int(video_fps / fps_sample))
        sample_indices = list(range(0, total_frames, frame_interval))

        print(f"[Preload] Video: {duration:.1f}s, {total_frames} frames, "
              f"sampling {len(sample_indices)} keyframes ({fps_sample}/s)")

        # ── Extract frames ────────────────────────────────────────────
        frames: List[tuple] = []  # (timestamp, PIL.Image)
        for target_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            # Resize for speed
            h, w = frame_bgr.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            timestamp = target_idx / video_fps
            frames.append((timestamp, pil_frame))

        cap.release()

        # ── Run GDINO on each keyframe ────────────────────────────────
        keyframes = []
        for timestamp, pil_frame in frames:
            detections = self.detection_service.detect_multiple(
                pil_frame, clean_q, threshold=0.25, verify=False
            )
            tracks = [
                {"track_id": i + 1, "bbox": d["box"], "score": d["score"]}
                for i, d in enumerate(detections)
            ]
            keyframes.append({"time": round(timestamp, 3), "tracks": tracks})

        elapsed = time.time() - t0
        detected_count = sum(1 for kf in keyframes if kf["tracks"])
        print(f"[Preload] Done in {elapsed:.1f}s — "
              f"{detected_count}/{len(keyframes)} keyframes have detections")

        return {"keyframes": keyframes, "duration": round(duration, 3)}


tracking_service = TrackingService()
