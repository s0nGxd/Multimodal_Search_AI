import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

class CaptionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CaptionService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Initializing CaptionService...")
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading BLIP model: {self.model_name} on {self.device}")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

caption_service = CaptionService()
