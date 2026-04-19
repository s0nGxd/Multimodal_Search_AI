import torch
from PIL import Image


class CaptionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CaptionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _ensure_loaded(self):
        if self._initialized:
            return
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.model_name = "Salesforce/blip-image-captioning-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading BLIP model (lazy): {self.model_name} on {self.device}")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
        self.model.eval()
        self._initialized = True

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        self._ensure_loaded()
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


caption_service = CaptionService()
