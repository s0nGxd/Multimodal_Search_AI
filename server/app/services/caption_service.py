import os
import torch
from PIL import Image


class CaptionService:
    """Florence-2 Large — ingestion captioning + lazy phrase-grounding at search time.

    Roles in the pipeline
    ---------------------
    1. generate_caption()  — Runs at upload time (once per image/frame).
       Produces a rich 1-4 sentence description stored in LanceDB, enabling BM25 text search.

    2. ground_phrase()  — Runs lazily during search "Deep Analysis" mode on the top N candidates.
       Uses CAPTION_TO_PHRASE_GROUNDING to verify whether a specific phrase (adjective + action +
       object) can be physically located in the image.  Returns the best bounding box and a
       confidence score.  If no box is found the score is 0.0.

       This is the correct use of Florence-2 for descriptive search: it understands compound
       natural-language phrases like "brown dog running", "woman in red jacket", "small kitten
       sleeping on a sofa" and either finds them or says no.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CaptionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _ensure_loaded(self):
        if self._initialized:
            return
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.model_name = "microsoft/Florence-2-large"

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Loading Florence-2 (lazy): {self.model_name} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self._initialized = True

    def _run_task(self, image: Image.Image, task: str, text_input: str = None,
                  max_new_tokens: int = 256, num_beams: int = 3) -> dict:
        """Execute a single Florence-2 task and return the post-processed result dict."""
        dtype = getattr(self.model, "dtype", torch.float32)
        full_text = text_input if text_input else task
        inputs = self.processor(
            text=full_text, images=image, return_tensors="pt"
        ).to(self.device, dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=False,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        """Generate a rich, concise description for storage at ingestion time.

        Uses MORE_DETAILED_CAPTION (1-4 sentences).  The stored text is used by:
          - BM25 full-text search  (e.g. "brown dog" → directly finds the image)
          - caption_vector ANN search  (SigLIP-embedded description)
        """
        self._ensure_loaded()
        if image.mode != "RGB":
            image = image.convert("RGB")

        task = "<MORE_DETAILED_CAPTION>"
        result = self._run_task(image, task, max_new_tokens=512, num_beams=3)
        return str(result.get(task, "")).strip()

    @torch.no_grad()
    def ground_phrase(self, image: Image.Image, phrase: str) -> dict:
        """Verify whether *phrase* can be physically located in *image*.

        Uses Florence-2 CAPTION_TO_PHRASE_GROUNDING — the model reads the phrase as a
        natural-language query (e.g. "brown dog running on grass") and tries to produce a
        bounding box that exactly matches it.  Unlike Grounding DINO, Florence-2 is a
        generative model that understands the *full semantic meaning* of the phrase rather
        than decomposing it into token matches, giving it far superior adjective + action +
        object comprehension.

        Returns
        -------
        dict with keys:
          - "found"  : bool   — True if at least one box was grounded
          - "score"  : float  — confidence 0.0–1.0 (0.0 if not found)
          - "box"    : list[float] | None  — normalized [x1, y1, x2, y2] of best box

        Latency: ~2–4 s per image on GPU.  Cap callers to 8 images max (~30 s total).
        """
        self._ensure_loaded()
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Florence-2 expects the caption to align with the text_input for grounding
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        try:
            result = self._run_task(
                image, task,
                text_input=f"{task}{phrase}",
                max_new_tokens=512, num_beams=3
            )
        except Exception as e:
            print(f"Florence-2 grounding error: {e}")
            return {"found": False, "score": 0.0, "box": None}

        grounding = result.get(task, {})
        bboxes = grounding.get("bboxes", [])
        labels = grounding.get("labels", [])

        if not bboxes:
            return {"found": False, "score": 0.0, "box": None}

        # Look for the label that best matches the phrase — pick the first grounded box
        # (Florence-2 grounding outputs are naturally ordered by relevance)
        w, h = image.size
        best_box = None
        best_score = 0.0
        phrase_lower = phrase.lower()

        for i, (box, label) in enumerate(zip(bboxes, labels)):
            label_lower = str(label).lower()
            # Simple word overlap score — how much of the phrase appears in the label
            phrase_words = set(phrase_lower.split())
            label_words = set(label_lower.split())
            overlap = len(phrase_words & label_words) / max(len(phrase_words), 1)
            # Earlier boxes score slightly higher (Florence orders by confidence)
            position_bonus = max(0.0, 0.1 - i * 0.01)
            score = min(1.0, 0.60 + (overlap * 0.35) + position_bonus)

            if score > best_score:
                best_score = score
                # Normalise absolute coords → 0..1
                x1, y1, x2, y2 = box
                best_box = [x1 / w, y1 / h, x2 / w, y2 / h]

        if best_box is None:
            return {"found": False, "score": 0.0, "box": None}

        return {"found": True, "score": best_score, "box": best_box}


caption_service = CaptionService()
