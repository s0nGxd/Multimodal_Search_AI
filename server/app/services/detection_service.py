import torch
from PIL import Image
from ultralytics import YOLOWorld


class DetectionService:
    """YOLO-World — used for open-vocabulary bounding box detection and video tracking.

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
        print("Initializing DetectionService (YOLO-World)...")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load YOLO-World Large (best accuracy for open-vocabulary detection)
        self.yolo_world = YOLOWorld("yolov8l-world.pt")
        self.yolo_world.to(self.device)

    def _query_to_text(self, query: str) -> str:
        # YOLO-World just takes the phrase directly
        return query.strip().lower()

    @torch.no_grad()
    def detect(self, image: Image.Image, query: str, threshold: float = 0.30,
               verify: bool = False):
        """Returns the single highest-confidence box for the query phrase."""
        text = self._query_to_text(query)
        self.yolo_world.set_classes([text])
        
        # Run YOLO-World
        results = self.yolo_world(image, conf=threshold, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return None
            
        boxes = results[0].boxes
        
        # Get the highest confidence box
        best_idx = torch.argmax(boxes.conf).item()
        raw_box = boxes.xyxy[best_idx].tolist()
        score = boxes.conf[best_idx].item()
        
        # Crop verification — only for static images
        if verify and not self._verify_crop(image, raw_box, query):
            return None

        width, height = image.size
        normalized_box = [
            raw_box[0] / width,
            raw_box[1] / height,
            raw_box[2] / width,
            raw_box[3] / height
        ]

        return {
            "box": normalized_box,
            "score": float(score),
            "label": text
        }

    @torch.no_grad()
    def detect_multiple(self, image: Image.Image, query: str,
                        threshold: float = 0.30, verify: bool = False):
        """Returns all boxes above threshold using YOLO-World."""
        text = self._query_to_text(query)
        self.yolo_world.set_classes([text])
        
        # Run YOLO-World
        results = self.yolo_world(image, conf=threshold, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return []
            
        boxes = results[0].boxes
        detections = []
        
        width, height = image.size
        
        for i in range(len(boxes)):
            raw_box = boxes.xyxy[i].tolist()
            score = boxes.conf[i].item()

            if verify and not self._verify_crop(image, raw_box, query):
                continue

            norm_box = [
                raw_box[0] / width,
                raw_box[1] / height,
                raw_box[2] / width,
                raw_box[3] / height
            ]
            detections.append({
                "box": norm_box,
                "score": float(score),
                "label": text
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

    def detect_dense(self, image: Image.Image) -> list:
        """Finds all distinct objects in an image using YOLO-World + SAHI.
        Returns:
            list of dicts: [{"bbox": [x1, y1, x2, y2], "label": "person", "score": 0.8}, ...]
        """
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        
        # We only care about objects relevant to a multimodal search index.
        # This prevents the database from being flooded with useless noise.
        ALLOWED_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 
            'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'sports ball', 
            'kite', 'skateboard', 'surfboard', 'tennis racket', 'bottle'
        ]

        if not hasattr(self, 'sahi_model'):
            # Use YOLO-World Large for surveillance-grade detection
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path='yolov8l-world.pt',
                confidence_threshold=0.25,
                device=self.device
            )
            # Set the classes for YOLO-World
            self.sahi_model.model.set_classes(ALLOWED_CLASSES)

        # Slice the image into tiles, run YOLO on tiles + full image, and merge.
        # perform_standard_pred=True ensures large objects (like the black dog) 
        # are detected as a whole on the full image pass.
        result = get_sliced_prediction(
            image,
            self.sahi_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=True,
            postprocess_type="NMM", # Non-Max Merging is better for sliced inference
            postprocess_match_threshold=0.5
        )
        
        ALLOWED_CLASSES_SET = set(ALLOWED_CLASSES)
        
        # Deduplicate and normalize boxes
        final_boxes = []
        
        for pred in result.object_prediction_list:
            cls_name = pred.category.name
            if cls_name in ALLOWED_CLASSES_SET:
                b = pred.bbox.to_xyxy() # [x1, y1, x2, y2]
                final_boxes.append({
                    "bbox": b,
                    "label": cls_name,
                    "score": pred.score.value
                })
                
        # Limit to the top 40 most confident objects per frame to prevent DB bloat
        final_boxes.sort(key=lambda x: x["score"], reverse=True)
        return final_boxes[:40]


detection_service = DetectionService()
