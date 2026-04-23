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
            
        # Stateful memory for real-time tracking
        self.active_tracks = []
        self.lost_tracks = []
        self.next_track_id = 1
        self.last_video_id = None

    # ── Single-frame detection (for images + legacy fallback) ────────────

    def detect_and_track(self, frame: Image.Image, query: str,
                         return_all_detections: bool = False) -> List[Dict]:
        """Stateful single-frame detection and tracking."""
        clean_q = _clean_query_for_gdino(query)
        
        # threshold 0.25 is more lenient for real-time
        detections = self.detection_service.detect_multiple(
            frame, clean_q, threshold=0.25, verify=False
        )
        # Limit to top 3
        detections = detections[:3]
        
        current_tracks = []
        matched_det_indices = set()
        
        # 1. Match active tracks
        for active_t in self.active_tracks:
            best_iou, best_idx = 0.25, -1
            for i, det in enumerate(detections):
                if i in matched_det_indices: continue
                boxA, boxB = active_t["bbox"], det["box"]
                xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
                xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                if inter > 0:
                    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
                    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
                    iou = inter / float(areaA + areaB - inter + 1e-6)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
            if best_idx != -1:
                matched_det_indices.add(best_idx)
                current_tracks.append({"track_id": active_t["track_id"], "bbox": detections[best_idx]["box"], "score": detections[best_idx]["score"]})

        # 2. Recovery from lost
        new_lost = []
        for lost_t in self.lost_tracks:
            if lost_t.get("age", 0) > 10: continue # Keep lost for ~10 frames in real-time
            best_iou, best_idx = 0.15, -1
            for i, det in enumerate(detections):
                if i in matched_det_indices: continue
                boxA, boxB = lost_t["bbox"], det["box"]
                xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
                xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                if inter > 0:
                    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
                    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
                    iou = inter / float(areaA + areaB - inter + 1e-6)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
            if best_idx != -1:
                matched_det_indices.add(best_idx)
                current_tracks.append({"track_id": lost_t["track_id"], "bbox": detections[best_idx]["box"], "score": detections[best_idx]["score"]})
            else:
                lost_t["age"] = lost_t.get("age", 0) + 1
                new_lost.append(lost_t)

        # 3. Demote unmatched active to lost
        matched_ids = {t["track_id"] for t in current_tracks}
        for active_t in self.active_tracks:
            if active_t["track_id"] not in matched_ids:
                active_t["age"] = 1
                new_lost.append(active_t)

        # 4. New tracks
        for i, det in enumerate(detections):
            if i not in matched_det_indices and det["score"] >= 0.40:
                current_tracks.append({"track_id": self.next_track_id, "bbox": det["box"], "score": det["score"]})
                self.next_track_id += 1
                
        self.active_tracks = current_tracks
        self.lost_tracks = new_lost
        return current_tracks

    def reset_for_new_video(self, video_id: str, query: str):
        """Resets tracking state if we've switched videos or queries."""
        if video_id != self.last_video_id:
            self.active_tracks = []
            self.lost_tracks = []
            self.next_track_id = 1
            self.last_video_id = video_id

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
        next_track_id = 1
        active_tracks = [] # list of obj: {"track_id": 1, "bbox": [..]}
        lost_tracks = []   # Tracks not seen in current frame, but kept for recovery

        for timestamp, pil_frame in frames:
            detections = self.detection_service.detect_multiple(
                pil_frame, clean_q, threshold=0.25, verify=False
            )
            
            # Limit to top 3 detections per frame to prevent noise
            detections = detections[:3]
            
            # Simple IoU tracker
            current_tracks = []
            matched_det_indices = set()
            
            # 1. Try to match each ACTIVE track to a new detection (IoU)
            for active_t in active_tracks:
                best_iou = 0.25
                best_det_idx = -1
                for i, det in enumerate(detections):
                    if i in matched_det_indices: continue
                    
                    boxA, boxB = active_t["bbox"], det["box"]
                    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
                    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    if inter > 0:
                        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
                        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
                        iou = inter / float(areaA + areaB - inter + 1e-6)
                        if iou > best_iou:
                            best_iou, best_det_idx = iou, i
                
                if best_det_idx != -1:
                    matched_det_indices.add(best_det_idx)
                    current_tracks.append({
                        "track_id": active_t["track_id"],
                        "bbox": detections[best_det_idx]["box"],
                        "score": detections[best_det_idx]["score"]
                    })
            
            # 2. Recovery: Try to match LOST tracks to remaining detections
            # This handles objects reappearing after occlusion
            new_lost_tracks = []
            for lost_t in lost_tracks:
                # If track has been lost for more than 4 sample frames (~4s), drop it
                if lost_t.get("age", 0) > 4:
                    continue
                
                best_iou = 0.15 # Lower threshold for recovery
                best_det_idx = -1
                for i, det in enumerate(detections):
                    if i in matched_det_indices: continue
                    # (Standard IoU calculation...)
                    boxA, boxB = lost_t["bbox"], det["box"]
                    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
                    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    if inter > 0:
                        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
                        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
                        iou = inter / float(areaA + areaB - inter + 1e-6)
                        if iou > best_iou:
                            best_iou, best_det_idx = iou, i
                
                if best_det_idx != -1:
                    matched_det_indices.add(best_det_idx)
                    current_tracks.append({
                        "track_id": lost_t["track_id"],
                        "bbox": detections[best_det_idx]["box"],
                        "score": detections[best_det_idx]["score"]
                    })
                else:
                    # Still lost, increment age and keep in buffer
                    lost_t["age"] = lost_t.get("age", 0) + 1
                    new_lost_tracks.append(lost_t)

            # 3. Handle unmatched active tracks (move them to lost)
            matched_ids = {t["track_id"] for t in current_tracks}
            for active_t in active_tracks:
                if active_t["track_id"] not in matched_ids:
                    active_t["age"] = 1
                    new_lost_tracks.append(active_t)

            # 4. Handle new detections (spawn new track)
            for i, det in enumerate(detections):
                if i not in matched_det_indices and det["score"] >= 0.40:
                    current_tracks.append({
                        "track_id": next_track_id,
                        "bbox": det["box"],
                        "score": det["score"]
                    })
                    next_track_id += 1
            
            active_tracks = current_tracks
            lost_tracks = new_lost_tracks
            keyframes.append({"time": round(timestamp, 3), "tracks": current_tracks})

        elapsed = time.time() - t0
        detected_count = sum(1 for kf in keyframes if kf["tracks"])
        print(f"[Preload] Done in {elapsed:.1f}s — "
              f"{detected_count}/{len(keyframes)} keyframes have detections")

        return {"keyframes": keyframes, "duration": round(duration, 3)}

tracking_service = TrackingService()
