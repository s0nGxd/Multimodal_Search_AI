import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image

try:
    from norfair import Detection, Tracker
except ImportError:
    Detection, Tracker = None, None

def robust_hybrid_distance(detection: Detection, tracked_object) -> float:
    try:
        det = detection.points[0]
        trk = tracked_object.estimate[0]
        
        # Calculate Centroid Distance
        det_c = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
        trk_c = [(trk[0] + trk[2]) / 2, (trk[1] + trk[3]) / 2]
        dist = np.sqrt((det_c[0] - trk_c[0])**2 + (det_c[1] - trk_c[1])**2)
        
        return min(1.0, dist / 1200.0)
    except:
        return 1.0

class TrackingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrackingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from app.services.detection_service import detection_service
        self.detection_service = detection_service
        
        self.tracker = Tracker(
            distance_function=robust_hybrid_distance,
            distance_threshold=0.95, 
            initialization_delay=0,
            hit_counter_max=40,
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
            self.tracker = Tracker(
                distance_function=robust_hybrid_distance,
                distance_threshold=0.95,
                initialization_delay=0,
                hit_counter_max=40,
                past_detections_length=5
            )
            self.current_video_id = norm_id
            self.current_query = query
    
    def detect_and_track(self, frame: Image.Image, query: str, return_all_detections: bool = False) -> List[Dict]:
        width, height = frame.size
        detections_raw = self._detect_multiple(frame, query)
        
        norfair_detections = []
        for det in detections_raw:
            box = det["box"]
            points = np.array([[box[0]*width, box[1]*height, box[2]*width, box[3]*height]], dtype=np.float32)
            norfair_detections.append(Detection(points=points, scores=np.array([det["score"]])))
        
        try:
            tracked_objects = self.tracker.update(detections=norfair_detections)
            results = []
            for obj in tracked_objects:
                if obj.hit_counter < 1: continue

                est = obj.estimate[0]
                normalized_box = [
                    float(np.clip(est[0]/width, 0, 1)), float(np.clip(est[1]/height, 0, 1)),
                    float(np.clip(est[2]/width, 0, 1)), float(np.clip(est[3]/height, 0, 1))
                ]
                
                # FIXED: Removed 'w > 0.9' filter to allow for large objects like mountains
                w = normalized_box[2] - normalized_box[0]
                if w < 0.005: continue 

                score = float(obj.last_detection.scores[0]) if obj.last_detection else 0.6
                results.append({"track_id": obj.id, "bbox": normalized_box, "score": score})
            return results
        except Exception:
            return []
    
    def _detect_multiple(self, frame: Image.Image, query: str, max_detections: int = 5) -> List[Dict]:
        width, height = frame.size
        # Added 'mountain' specific keywords to improve background recall
        texts = [[f"a photo of a {query}", f"{query}", f"the {query} in the background"]]
        try:
            processor = self.detection_service.processor
            model = self.detection_service.model
            inputs = processor(text=texts, images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            target_sizes = torch.tensor([[height, width]]).to(self.device)
            results = processor.image_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.10 # Lowered for background objects
            )
            
            i = 0
            boxes, scores = results[i]["boxes"], results[i]["scores"]
            if len(scores) == 0: return []
            
            from torchvision.ops import nms
            keep = nms(boxes, scores, iou_threshold=0.3)
            
            boxes = boxes[keep][:max_detections]
            scores = scores[keep][:max_detections]
            
            detections = []
            for b, s in zip(boxes, scores):
                box = b.tolist()
                detections.append({
                    "box": [box[0]/width, box[1]/height, box[2]/width, box[3]/height],
                    "score": s.item()
                })
            return detections
        except Exception:
            return []

    def process_video_frames(self, frames: List[Image.Image], query: str) -> List[List[Dict]]:
        return [self.detect_and_track(f, query) for f in frames]

tracking_service = TrackingService()
