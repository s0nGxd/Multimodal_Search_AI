import os
import lancedb
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class SearchService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Initializing SearchService...")
        self.db_uri = os.getenv("LANCEDB_URI", "./lancedb_db")
        self.model_name = os.getenv("EMBED_MODEL", "openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Model
        print(f"Loading CLIP model: {self.model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()

        # Connect to DB
        self.db = lancedb.connect(self.db_uri)
        try:
            self.table = self.db.open_table("images")
        except:
            print("Table 'images' not found. It will need to be created via ingestion.")
            self.table = None

    def refresh_table(self):
        """Re-opens the table in case it was created/updated."""
        try:
            self.table = self.db.open_table("images")
        except:
            self.table = None

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype("float32")[0]

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype("float32")[0]

    def search(self, query: str, k: int = 20, threshold: float = 0.5):
        if self.table is None:
            self.refresh_table()
            if self.table is None:
                return []

        query_vec = self.embed_text(query)
        
        # Perform search
        # We select _distance to get the cosine distance
        results = (
            self.table.search(query_vec)
            .metric("cosine")
            .limit(k)
            .select(["photo_id", "photo_image_url", "description"])
            .to_pandas()
        )
        
        if results.empty:
            return []
            
        # Filter by threshold (cosine distance: lower is better)
        # Note: LanceDB returns distance as '_distance'
        if "_distance" in results.columns:
            results = results[results["_distance"] <= threshold]

        return results.to_dict(orient="records")

search_service = SearchService()
