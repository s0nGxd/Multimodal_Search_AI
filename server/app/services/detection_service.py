import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class DetectionService:
    """Grounding DINO — used for bounding box detection and video tracking.

    _verify_crop is available but ONLY used for static image detection
    (when the user clicks a search result).  It is NEVER used during
    video tracking because it runs a full SigLIP forward pass per crop,
    which is far too expensive for real-time use (~500ms per box).
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DetectionService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Initializing DetectionService (Grounding DINO)...")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model_name = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()

    def _query_to_text(self, query: str) -> str:
        """Grounding DINO expects a phrase ending with a period."""
        return f"{query.strip().rstrip('.')}."

    @torch.no_grad()
    def detect(self, image: Image.Image, query: str, threshold: float = 0.30,
               verify: bool = False):
        """Returns the single highest-confidence box for the query phrase.

        Parameters
        ----------
        verify : bool
            If True, runs SigLIP crop verification to reject false positives.
            Should be True for static image detection, False for video tracking.
        """
        text = self._query_to_text(query)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        width, height = image.size
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            target_sizes=[(height, width)]
        )

        scores = results[0]["scores"]
        if len(scores) == 0:
            return None

        best_idx = scores.argmax().item()
        raw_box = results[0]["boxes"][best_idx].tolist()

        # Crop verification — only for static images (search result clicks),
        # NEVER for real-time video tracking.
        if verify and not self._verify_crop(image, raw_box, query):
            return None

        normalized_box = [
            raw_box[0] / width,
            raw_box[1] / height,
            raw_box[2] / width,
            raw_box[3] / height
        ]

        labels = results[0].get("labels", [])
        label = labels[best_idx] if best_idx < len(labels) else query

        return {
            "box": normalized_box,
            "score": float(scores[best_idx].item()),
            "label": label
        }

    @torch.no_grad()
    def detect_multiple(self, image: Image.Image, query: str,
                        threshold: float = 0.30, verify: bool = False):
        """Returns all boxes above threshold.

        Parameters
        ----------
        verify : bool
            If True, runs SigLIP crop verification on each box.
            Should be False for video tracking (performance).
        """
        text = self._query_to_text(query)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        width, height = image.size
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            target_sizes=[(height, width)]
        )

        scores = results[0]["scores"]
        if len(scores) == 0:
            return []

        boxes = results[0]["boxes"]
        labels = results[0].get("labels", [])

        detections = []
        for i in range(len(scores)):
            raw_box = boxes[i].tolist()

            if verify and not self._verify_crop(image, raw_box, query):
                continue

            label = labels[i] if i < len(labels) else query
            norm_box = [
                raw_box[0] / width,
                raw_box[1] / height,
                raw_box[2] / width,
                raw_box[3] / height
            ]
            detections.append({
                "box": norm_box,
                "score": float(scores[i].item()),
                "label": label
            })

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    def _verify_crop(self, image: Image.Image, box_coords: list, query: str) -> bool:
        """SigLIP crop verification — expensive, only for static image detection."""
        try:
            from app.services.search_service import search_service

            width, height = image.size
            x1 = max(0, box_coords[0] - 10)
            y1 = max(0, box_coords[1] - 10)
            x2 = min(width, box_coords[2] + 10)
            y2 = min(height, box_coords[3] + 10)

            crop = image.crop((x1, y1, x2, y2))

            prompts = [
                f"a photo of a {query}",
                "a photo of a different animal or person",
                "a photo of a generic background or object"
            ]

            t_txt = search_service.processor(text=prompts, return_tensors='pt', padding=True).to(search_service.device)
            t_img = search_service.processor(images=crop, return_tensors='pt').to(search_service.device)

            with torch.no_grad():
                text_feat = search_service.model.get_text_features(**t_txt)
                text_feat /= text_feat.norm(p=2, dim=-1, keepdim=True)
                img_feat = search_service.model.get_image_features(**t_img)
                img_feat /= img_feat.norm(p=2, dim=-1, keepdim=True)
                scale = search_service.model.logit_scale.exp().item()
                bias = search_service.model.logit_bias.item()

            sims = (img_feat @ text_feat.T) * scale + bias
            probs = torch.softmax(sims, dim=-1)[0]
            return probs[0].item() >= 0.20

        except Exception as e:
            print(f"Crop verification error: {e}")
            return True


detection_service = DetectionService()
