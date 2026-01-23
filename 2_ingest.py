import os
import lancedb
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# 1. Setup - Models, Locating Files, etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

db = lancedb.connect("./lancedb_data")
IMAGE_FOLDER = "images"

def get_embedding(image_path):
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # Normalize
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()
    except:
        return None

# 2. Process - give embeddings to each images
data = []
print("Generating embeddings...")

files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg')]

for filename in files:
    path = os.path.join(IMAGE_FOLDER, filename)
    vector = get_embedding(path)
    if vector:
        data.append({"filename": filename, "path": path, "vector": vector})

# 3. Save into LanceDB
if data:
    db.create_table("my_images", data=data, mode="overwrite")
    print(f"Success! {len(data)} images indexed in LanceDB.")
else:
    print("No images found. Do you have an image dataset?")