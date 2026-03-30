import torch
import io
import requests
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class DetectionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DetectionService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Initializing DetectionService (OWL-ViT)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "google/owlvit-base-patch32"
        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: Image.Image, query: str):
        # We query with multiple contextual strings to improve recall for partial objects
        texts = [[f"a photo of a {query}", f"{query}", f"a {query}"]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        # Get target sizes
        width, height = image.size
        
        target_sizes = torch.tensor([[height, width]]).to(self.device)
        # Lowered threshold to 0.05 to catch obscured/truncated targets
        results = self.processor.image_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.05)
        
        i = 0  # Retrieve predictions for the first image
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
        if len(scores) == 0:
            return None
            
        # Get the box with the highest score
        best_idx = scores.argmax().item()
        raw_box = boxes[best_idx].tolist()
        
        # Normalize to 0..1
        normalized_box = [
            raw_box[0] / width,
            raw_box[1] / height,
            raw_box[2] / width,
            raw_box[3] / height
        ]
        
        score = scores[best_idx].item()
        
        return {
            "box": normalized_box, # [xmin, ymin, xmax, ymax]
            "score": score
        }

detection_service = DetectionService()
