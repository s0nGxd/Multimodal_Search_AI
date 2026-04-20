import os
import re
import shutil
import tempfile
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image

try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 is not installed. Falling back to frame-by-frame Grounding DINO for tracking.")

# ── Same modifier set used by search — keep in sync ──────────────────────
_FILLER = {
    'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'for',
    'is', 'to', 'by', 'from', 'its', 'this', 'that',
    'color', 'colour', 'colored', 'coloured', 'looking', 'like', 'type',
    'kind', 'style', 'very', 'really', 'quite', 'pretty', 'beautiful',
}


def _clean_query_for_gdino(query: str) -> str:
    """Simplify a user query into a phrase GDINO can understand.

    'brown color dog running' → 'brown dog running'
    'small black cat'         → 'small black cat' (unchanged, all useful)
    'a photo of a large dog'  → 'large dog'

    GDINO handles simple adjective-noun phrases well; it's the
    filler words like 'color', 'type', 'looking' that confuse it.
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
        
        # We lazy load the predictor to avoid memory overhead unless tracking is requested.
        self.predictor = None
        self.sam2_checkpoint = os.getenv("SAM2_CHECKPOINT", None) # Optional local path
        if not self.sam2_checkpoint and SAM2_AVAILABLE:
            self.sam2_cfg = "sam2_hiera_s.yaml"

    def _init_predictor(self):
        if not SAM2_AVAILABLE:
            return False
            
        if self.predictor is not None:
            return True
            
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id="facebook/sam2-hiera-small", filename="sam2_hiera_small.pt")
            self.predictor = build_sam2_video_predictor(self.sam2_cfg, ckpt_path, device=self.device)
            return True
        except Exception as e:
            print(f"Failed to load SAM 2 Predictor: {e}")
            return False

    def detect_and_track(self, frame: Image.Image, query: str, return_all_detections: bool = False) -> List[Dict]:
        """Single-frame detection. Returns all matching objects when return_all_detections=True.
        
        Cleans the query to strip filler words like 'color', 'type', etc.
        that confuse Grounding DINO.  verify=False for real-time performance.
        """
        clean_q = _clean_query_for_gdino(query)
        
        if return_all_detections:
            detections = self.detection_service.detect_multiple(
                frame, clean_q, threshold=0.20, verify=False
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
        """No-op for the new architecture as SAM2 is stateless across API calls."""
        pass

    def process_video_frames(self, frames: List[Image.Image], query: str) -> List[List[Dict]]:
        """Processes the entire video. Uses SAM 2 by default with Grounding DINO fallback."""
        clean_q = _clean_query_for_gdino(query)
        
        has_sam2 = self._init_predictor()
        
        # FALLBACK MODE: Frame-by-frame Grounding DINO
        if not has_sam2:
            print("Running in Grounding DINO fallback mode (SAM 2 unavailable).")
            results = []
            for frame in frames:
                det = self.detection_service.detect(frame, clean_q, verify=False)
                if det and det["score"] > 0.15:
                    results.append([{"track_id": 1, "bbox": det["box"], "score": det["score"]}])
                else:
                    results.append([])
            return results
            
        # SAM 2 MODE
        tmpdir = tempfile.mkdtemp()
        try:
            for i, frame in enumerate(frames):
                frame_path = os.path.join(tmpdir, f"{i:05d}.jpg")
                frame.convert("RGB").save(frame_path)
                
            inference_state = self.predictor.init_state(video_path=tmpdir)
            
            first_box_det = self.detection_service.detect(frames[0], clean_q, verify=False)
            results = [[] for _ in range(len(frames))]
            
            if not first_box_det:
                current_frame_idx = 0
                obj_found = False
                for i in range(1, len(frames)):
                    det = self.detection_service.detect(frames[i], clean_q, verify=False)
                    if det and det["score"] > 0.15:
                        first_box_det = det
                        current_frame_idx = i
                        obj_found = True
                        break
                if not obj_found:
                    return results
            else:
                current_frame_idx = 0
                
            width, height = frames[0].size
            active_obj_id = 1
            
            def to_absolute(norm_box):
                return np.array([
                    norm_box[0] * width,
                    norm_box[1] * height,
                    norm_box[2] * width,
                    norm_box[3] * height
                ], dtype=np.float32)
                
            def mask_to_box(mask_np):
                if mask_np.sum() == 0:
                    return None
                y_indices, x_indices = np.where(mask_np > 0)
                if len(y_indices) == 0:
                    return None
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                return [
                    float(x_min) / width, float(y_min) / height, 
                    float(x_max) / width, float(y_max) / height
                ]

            abs_box = to_absolute(first_box_det["box"])
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=current_frame_idx,
                obj_id=active_obj_id,
                box=abs_box
            )
            
            latest_bbox = first_box_det["box"]
            results[current_frame_idx].append({"track_id": active_obj_id, "bbox": latest_bbox, "score": 1.0})
            
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, start_frame_idx=current_frame_idx):
                if out_frame_idx == current_frame_idx:
                    continue
                
                if active_obj_id in out_obj_ids:
                    idx = out_obj_ids.index(active_obj_id)
                    mask = (out_mask_logits[idx] > 0.0).cpu().numpy().squeeze()
                    bbox = mask_to_box(mask)
                    
                    if bbox is not None:
                        results[out_frame_idx].append({"track_id": active_obj_id, "bbox": bbox, "score": 1.0})
                        latest_bbox = bbox
                    else:
                        re_det = self.detection_service.detect(frames[out_frame_idx], clean_q, verify=False)
                        if re_det and re_det["score"] > 0.15:
                            active_obj_id += 1
                            abs_box_re = to_absolute(re_det["box"])
                            self.predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=active_obj_id,
                                box=abs_box_re
                            )
                            results[out_frame_idx].append({"track_id": active_obj_id, "bbox": re_det["box"], "score": 1.0})
                            
        except Exception as e:
            print(f"Tracking error: {e}")
        finally:
            shutil.rmtree(tmpdir)
            
        return results

tracking_service = TrackingService()
