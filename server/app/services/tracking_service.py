import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import traceback

try:
    from norfair import Detection, Tracker
except ImportError:
    Detection, Tracker = None, None

def hybrid_distance(detection: Detection, tracked_object) -> float:
    """
    Hybrid distance: Robust IoU + Scaled Centroid distance.
    """
    try:
        det_box = detection.points[0] # [xmin, ymin, xmax, ymax]
        track_box = tracked_object.estimate[0]
        
        # 1. Calculate IoU
        ixmin = max(det_box[0], track_box[0])
        iymin = max(det_box[1], track_box[1])
        ixmax = min(det_box[2], track_box[2])
        iymax = min(det_box[3], track_box[3])
        iw = max(0, ixmax - ixmin)
        ih = max(0, iymax - iymin)
        area_intersection = iw * ih
        area_det = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
        area_track = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
        area_union = area_det + area_track - area_intersection
        iou = area_intersection / area_union if area_union > 0 else 0
        
        # 2. Calculate Centroid Distance
        det_center = [(det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2]
        track_center = [(track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2]
        dist = np.sqrt((det_center[0] - track_center[0])**2 + (det_center[1] - track_center[1])**2)
        
        # We prioritize IoU. If IoU is low or 0, we use centroid distance.
        # dist is in pixels. 500.0 is a reasonable normalization for 1080p/4k scaled frames.
        if iou > 0.1:
            return 1.0 - iou
        else:
            return min(0.95, dist / 500.0) 
            
    except Exception:
        return 1.0

class TrackingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrackingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        print("Initializing TrackingService (OWL-ViT + Norfair Optimized)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        from app.services.detection_service import detection_service
        self.detection_service = detection_service
        
        self.tracker = Tracker(
            distance_function=hybrid_distance,
            distance_threshold=0.9, # Permissive: allows for faster movement between detections
            initialization_delay=0,
            hit_counter_max=100,    # Robust: wait longer for object to reappear
            past_detections_length=5
        )
        
        self.current_video_id: Optional[str] = None
        self.current_query: Optional[str] = None

    def _normalize_id(self, video_id: Optional[str]) -> Optional[str]:
        if not video_id: return None
        return video_id.split("/")[-1].split("?")[0]
    
    def reset_for_new_video(self, video_id: str, query: str = ""):
        norm_id = self._normalize_id(video_id)
        if self.current_video_id != norm_id or self.current_query != query:
            print(f"Resetting tracker for: Video={norm_id}, Query={query}")
            self.tracker = Tracker(
                distance_function=hybrid_distance,
                distance_threshold=0.9,
                initialization_delay=0,
                hit_counter_max=100,
                past_detections_length=5
            )
            self.current_video_id = norm_id
            self.current_query = query
    
    def detect_and_track(self, frame: Image.Image, query: str, return_all_detections: bool = False) -> List[Dict]:
        if Tracker is None:
            detections = self._detect_multiple(frame, query)
            return [{"track_id": i, "bbox": d["box"], "score": d["score"]} for i, d in enumerate(detections)]

        width, height = frame.size
        detections_raw = self._detect_multiple(frame, query)
        
        norfair_detections = []
        for det in detections_raw:
            box = det["box"]
            points = np.array([[box[0] * width, box[1] * height, box[2] * width, box[3] * height]], dtype=np.float32)
            norfair_detections.append(Detection(points=points, scores=np.array([det["score"]])))
        
        try:
            tracked_objects = self.tracker.update(detections=norfair_detections)
            results = []
            for obj in tracked_objects:
                est = obj.estimate[0]
                normalized_box = [
                    float(np.clip(est[0] / width, 0, 1)),
                    float(np.clip(est[1] / height, 0, 1)),
                    float(np.clip(est[2] / width, 0, 1)),
                    float(np.clip(est[3] / height, 0, 1))
                ]
                
                # Filter out invalid boxes
                if (normalized_box[2] - normalized_box[0]) < 0.005 or (normalized_box[3] - normalized_box[1]) < 0.005:
                    continue

                score = 0.8
                if obj.last_detection is not None:
                    score = float(obj.last_detection.scores[0])
                
                results.append({"track_id": obj.id, "bbox": normalized_box, "score": score})
            return results
        except Exception as e:
            print(f"Tracker error: {e}")
            return [{"track_id": 999, "bbox": d["box"], "score": d["score"]} for d in detections_raw]
    
    def _detect_multiple(self, frame: Image.Image, query: str, max_detections: int = 5) -> List[Dict]:
        width, height = frame.size
        # Minimal query for speed
        texts = [[f"a photo of a {query}", f"{query}"]]
        
        try:
            processor = self.detection_service.processor
            model = self.detection_service.model
            
            inputs = processor(text=texts, images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            target_sizes = torch.tensor([[height, width]]).to(self.device)
            results = processor.image_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=0.15 # Better balance: catches blurry dog but ignores background noise
            )
            
            i = 0
            boxes, scores = results[i]["boxes"], results[i]["scores"]
            if len(scores) == 0: return []
            
            keep_idx = scores.argsort(descending=True)[:max_detections]
            detections = []
            for idx in keep_idx:
                box = boxes[idx].tolist()
                detections.append({
                    "box": [box[0]/width, box[1]/height, box[2]/width, box[3]/height],
                    "score": scores[idx].item()
                })
            return detections
        except Exception:
            return []

    def process_video_frames(self, frames: List[Image.Image], query: str) -> List[List[Dict]]:
        all_results = []
        for frame in frames:
            result = self.detect_and_track(frame, query)
            all_results.append(result)
        return all_results

tracking_service = TrackingService()
