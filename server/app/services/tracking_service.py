import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    score: float
    age: int = 0
    hits: int = 1
    
    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "score": self.score
        }


class ByteTracker:
    def __init__(
        self,
        track_thresh: float = 0.3,
        track_buffer: int = 30,
        match_thresh: float = 0.3,
        new_track_thresh: float = 0.4
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.new_track_thresh = new_track_thresh
        
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        self.frame_count = 0
        self.max_time_lost = track_buffer
    
    def reset(self):
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
    
    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections: List) -> List[Track]:
        self.frame_count += 1
        
        if not detections:
            self._remove_lost_tracks()
            return []
        
        try:
            high_det = [(d[0], d[1]) for d in detections if d[1] >= self.track_thresh]
            low_det = [(d[0], d[1]) for d in detections if self.new_track_thresh <= d[1] < self.track_thresh]
            
            self._activate_new_tracks(high_det)
            self._update_tracks(high_det)
            self._match_lost_tracks(low_det)
            self._remove_lost_tracks()
            
            return [t for t in self.tracks if t.age <= 1]
        except Exception as e:
            print(f"Tracker update error: {e}")
            self.tracks = []
            return []
    
    def _activate_new_tracks(self, detections: List):
        if not self.tracks:
            for bbox, score in detections:
                track = Track(
                    track_id=self.track_id_counter,
                    bbox=bbox,
                    score=score,
                    age=0
                )
                self.tracks.append(track)
                self.track_id_counter += 1
            return
        
        try:
            matched, unmatched = self._match_detections_to_tracks(detections)
            
            for det_idx in unmatched:
                if det_idx < len(detections):
                    track = Track(
                        track_id=self.track_id_counter,
                        bbox=detections[det_idx][0],
                        score=detections[det_idx][1],
                        age=0
                    )
                    self.tracks.append(track)
                    self.track_id_counter += 1
        except Exception as e:
            print(f"Activate new tracks error: {e}")
    
    def _update_tracks(self, detections: List):
        try:
            matched, _ = self._match_detections_to_tracks(detections)
            
            for det_idx, track_idx in matched:
                if track_idx < len(self.tracks):
                    self.tracks[track_idx].bbox = detections[det_idx][0]
                    self.tracks[track_idx].score = detections[det_idx][1]
                    self.tracks[track_idx].age = 0
                    self.tracks[track_idx].hits += 1
        except Exception as e:
            print(f"Update tracks error: {e}")
    
    def _match_lost_tracks(self, low_detections: List):
        if not low_detections:
            return
            
        lost_tracks = [t for t in self.tracks if t.age > self.max_time_lost]
        if not lost_tracks:
            return
        
        lost_bboxes = [t.bbox for t in lost_tracks]
        det_bboxes = [d[0] for d in low_detections]
        
        iou_matrix = np.zeros((len(lost_bboxes), len(det_bboxes)))
        for i, lb in enumerate(lost_bboxes):
            for j, db in enumerate(det_bboxes):
                iou_matrix[i, j] = self.compute_iou(lb, db)
        
        matched_pairs = []
        for _ in range(min(len(lost_tracks), len(low_detections))):
            max_iou = 0
            max_pair = None
            for i in range(len(lost_tracks)):
                for j in range(len(low_detections)):
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= self.match_thresh:
                        max_iou = iou_matrix[i, j]
                        max_pair = (i, j)
            
            if max_pair is None:
                break
                
            matched_pairs.append(max_pair)
            iou_matrix[max_pair[0], :] = 0
            iou_matrix[:, max_pair[1]] = 0
        
        for track_idx, det_idx in matched_pairs:
            lost_tracks[track_idx].bbox = low_detections[det_idx][0]
            lost_tracks[track_idx].score = low_detections[det_idx][1]
            lost_tracks[track_idx].age = 0
            self.tracks.append(lost_tracks[track_idx])
    
    def _match_detections_to_tracks(self, detections: List) -> tuple:
        if not self.tracks or not detections:
            return [], list(range(len(detections)))
        
        track_bboxes = [t.bbox for t in self.tracks]
        det_bboxes = [d[0] for d in detections]
        
        iou_matrix = np.zeros((len(track_bboxes), len(det_bboxes)))
        for i, tb in enumerate(track_bboxes):
            for j, db in enumerate(det_bboxes):
                iou_matrix[i, j] = self.compute_iou(tb, db)
        
        matched = []
        unmatched_dets = list(range(len(detections)))
        
        for _ in range(min(len(self.tracks), len(detections))):
            max_iou = 0
            max_pair = None
            for i in range(len(self.tracks)):
                for j in unmatched_dets:
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= self.match_thresh:
                        max_iou = iou_matrix[i, j]
                        max_pair = (i, j)
            
            if max_pair is None:
                break
                
            matched.append(max_pair)
            unmatched_dets.remove(max_pair[1])
            iou_matrix[max_pair[0], :] = 0
            iou_matrix[:, max_pair[1]] = 0
        
        return matched, unmatched_dets
    
    def _remove_lost_tracks(self):
        self.tracks = [t for t in self.tracks if t.age <= self.max_time_lost]
        for t in self.tracks:
            t.age += 1


class TrackingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrackingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        print("Initializing TrackingService (OWL-ViT + Custom ByteTrack)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        from app.services.detection_service import detection_service
        self.detection_service = detection_service
        
        self.tracker = ByteTracker(
            track_thresh=0.3,
            track_buffer=30,
            match_thresh=0.2,
            new_track_thresh=0.35
        )
        
        self.current_video_id: Optional[str] = None
    
    def reset_for_new_video(self, video_id: str):
        if self.current_video_id != video_id:
            self.tracker.reset()
            self.current_video_id = video_id
    
    def detect_and_track(self, frame, query: str, return_all_detections: bool = False) -> List[Dict]:
        from PIL import Image
        
        if isinstance(frame, Image.Image):
            width, height = frame.size
        else:
            return []
        
        detections = self._detect_multiple(frame, query)
        
        if not detections:
            return []
        
        try:
            det_tuples = [(d["box"], d["score"]) for d in detections]
            tracks = self.tracker.update(det_tuples)
            
            return [t.to_dict() for t in tracks]
        except Exception as e:
            print(f"Detect and track error: {e}")
            return []
    
    def _detect_multiple(self, frame, query: str, max_detections: int = 10) -> List[Dict]:
        from PIL import Image
        
        if isinstance(frame, Image.Image):
            width, height = frame.size
        else:
            return []
        
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
            print(f"Detection error: {e}")
            return []
    
    def process_video_frames(self, frames: List, query: str) -> List[List[Dict]]:
        all_results = []
        for frame in frames:
            result = self.detect_and_track(frame, query, return_all_detections=True)
            all_results.append(result)
        return all_results


tracking_service = TrackingService()