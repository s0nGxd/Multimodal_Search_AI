import os
import sys
from PIL import Image
import torch

# Add the current directory to sys.path to import app
sys.path.append(os.getcwd())

def test_dog_tracking():
    print("--- Starting Tracking Test: Brown Dog ---")
    
    try:
        from app.services.tracking_service import tracking_service
        from app.services.detection_service import detection_service
    except ImportError as e:
        print(f"Error importing services: {e}")
        print("Make sure you are running this from the 'server' directory.")
        return

    # 1. Setup paths
    image_path = os.path.join("data", "13070184_3840_2160_60fps_frame_0.jpg")
    video_id = "13070184_3840_2160_60fps.mp4"
    query = "brown dog"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 2. Load Image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    
    # 3. Reset Tracker for this specific video
    print(f"Resetting tracker for video: {video_id}")
    tracking_service.reset_for_new_video(video_id)

    # 4. Run Detection and Tracking
    print(f"Running OWL-ViT + Norfair for query: '{query}'...")
    results = tracking_service.detect_and_track(img, query)

    # 5. Output Results
    if not results:
        print("No tracks found. This might be due to the confidence threshold (0.15).")
        # Try raw detection to see if anything is detected at all
        print("Checking raw detection for debugging...")
        raw_det = detection_service.detect(img, query)
        if raw_det:
            print(f"Raw detection found something! Score: {raw_det['score']:.4f}")
            print(f"Box: {raw_det['box']}")
        else:
            print("Raw detection also found nothing.")
    else:
        print(f"Success! Found {len(results)} track(s):")
        for res in results:
            print(f" - Track ID: {res['track_id']}")
            print(f"   Score: {res['score']:.4f}")
            print(f"   BBox (normalized): {res['bbox']}")

if __name__ == "__main__":
    test_dog_tracking()
