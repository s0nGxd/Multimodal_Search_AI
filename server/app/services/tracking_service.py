import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
try:
    from norfair import Detection, Tracker
except ImportError:
    # Fallback for environment setup
    Detection, Tracker = None, None

def iou(detection: Detection, tracked_object) -> float:
    """
    Computes IoU between a detection and a tracked object.
    Norfair expects distance, so we return 1.0 - IoU.
    """
    # detection.points: [[xmin, ymin], [xmax, ymax]]
    # tracked_object.estimate: [[xmin, ymin], [xmax, ymax]]
    
    det_box = detection.points.flatten() # [xmin, ymin, xmax, ymax]
    track_box = tracked_object.estimate.flatten() # [xmin, ymin, xmax, ymax]
    
    # Calculate intersection
    ixmin = max(det_box[0], track_box[0])
    iymin = max(det_box[1], track_box[1])
    ixmax = min(det_box[2], track_box[2])
    iymax = min(det_box[3], track_box[3])
    
    iw = max(0, ixmax - ixmin)
    ih = max(0, iymax - iymin)
    area_intersection = iw * ih
    
    # Calculate union
    area_det = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
    area_track = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
    area_union = area_det + area_track - area_intersection
    
    if area_union <= 0:
        return 1.0
        
    iou_val = area_intersection / area_union
    return 1.0 - iou_val # Return distance (1 - IoU)

class TrackingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrackingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        print("Initializing TrackingService (OWL-ViT + Norfair)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        from app.services.detection_service import detection_service
        self.detection_service = detection_service
        
        self.tracker = Tracker(
            distance_function=iou,
            distance_threshold=0.7, # Max distance (1 - IoU) to consider a match
            initialization_delay=0,
            hit_counter_max=15,
            past_detections_length=5
        )
        
        self.current_video_id: Optional[str] = None
    
    def reset_for_new_video(self, video_id: str):
        if self.current_video_id != video_id:
            print(f"Resetting tracker for new video: {video_id}")
            self.tracker = Tracker(
                distance_function=iou,
                distance_threshold=0.7,
                initialization_delay=0,
                hit_counter_max=15,
                past_detections_length=5
            )
            self.current_video_id = video_id
    
    def detect_and_track(self, frame: Image.Image, query: str, return_all_detections: bool = False) -> List[Dict]:
        if Tracker is None:
            print("Norfair not installed, falling back to raw detection")
            detections = self._detect_multiple(frame, query)
            return [{"track_id": i, "bbox": d["box"], "score": d["score"]} for i, d in enumerate(detections)]

        width, height = frame.size
        detections_raw = self._detect_multiple(frame, query)
        
        norfair_detections = []
        for det in detections_raw:
            box = det["box"]
            # Convert to pixel coordinates for stability in distance calculation
            pixel_box = np.array([
                [box[0] * width, box[1] * height],
                [box[2] * width, box[3] * height]
            ])
            norfair_detections.append(Detection(points=pixel_box, scores=np.array([det["score"]])))
        
        tracked_objects = self.tracker.update(detections=norfair_detections)
        
        results = []
        for obj in tracked_objects:
            est = obj.estimate
            normalized_box = [
                float(est[0][0] / width),
                float(est[0][1] / height),
                float(est[1][0] / width),
                float(est[1][1] / height)
            ]
            
            score = 0.8
            if obj.last_detection is not None:
                score = float(obj.last_detection.scores[0])
            
            results.append({
                "track_id": obj.id,
                "bbox": normalized_box,
                "score": score
            })
            
        return results
    
    def _detect_multiple(self, frame: Image.Image, query: str, max_detections: int = 10) -> List[Dict]:
        width, height = frame.size
        texts = [[f"a photo of a {query}", f"{query}", f"a {query}", f"many {query}", f"all {query}"]]
        
        try:
            processor = self.detection_service.processor
            model = self.detection_service.model
            
            inputs = processor(text=texts, images=frame, return_tensors="pt").to(self.device)
            outputs = model(**inputs)
            
            target_sizes = torch.tensor([[height, width]]).to(self.device)
            results = processor.image_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=0.15
            )
            
            i = 0
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            
            if len(scores) == 0:
                return []
            
            keep_idx = scores.argsort(descending=True)[:max_detections]
            
            detections = []
            for idx in keep_idx:
                box = boxes[idx].tolist()
                score = scores[idx].item()
                
                normalized_box = [
                    box[0] / width,
                    box[1] / height,
                    box[2] / width,
                    box[3] / height
                ]
                
                detections.append({
                    "box": normalized_box,
                    "score": score,
                    "label": labels[idx].item()
                })
            
            return detections
            
        except Exception as e:
            print(f"Detection error in TrackingService: {e}")
            return []
    
    def process_video_frames(self, frames: List[Image.Image], query: str) -> List[List[Dict]]:
        all_results = []
        for frame in frames:
            result = self.detect_and_track(frame, query)
            all_results.append(result)
        return all_results

tracking_service = TrackingService()
