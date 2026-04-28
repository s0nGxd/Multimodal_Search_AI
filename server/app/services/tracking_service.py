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
        det_c = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
        trk_c = [(trk[0] + trk[2]) / 2, (trk[1] + trk[3]) / 2]
        dist = np.sqrt((det_c[0] - trk_c[0])**2 + (det_c[1] - trk_c[1])**2)
        # Search radius: 600px is strict enough to prevent drifting but handles fast dogs
        return min(1.0, dist / 600.0)
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
        pass
    
    def detect_and_track(self, frame: Image.Image, query: str, return_all_detections: bool = False) -> List[Dict]:
        return []
    
    def reset_for_new_video(self, video_id: str, query: str = ""):
        pass

    def process_video_frames(self, frames: List[Image.Image], query: str) -> List[List[Dict]]:
        return [[] for _ in frames]

tracking_service = TrackingService()
