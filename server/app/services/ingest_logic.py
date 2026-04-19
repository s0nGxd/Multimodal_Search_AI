import os
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
import lancedb
from lancedb.pydantic import LanceModel, Vector
import pandas as pd
import requests

# CONFIGURATION
n = 10  # number of images to process
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# 1. LOAD DATASET
print("=== LOADING DATASET ===")
# Ensure IDs are strings to match filenames later
df = pd.read_csv("photos.csv000", sep="\t", engine="python", on_bad_lines='skip')
df['photo_id'] = df['photo_id'].astype(str) 

# Create lookup dict for later
id_to_url = dict(zip(df.photo_id, df.photo_image_url))


# 2. DOWNLOAD IMAGES (this is the collecting images part of the gantt chart)
print("\n=== DOWNLOADING IMAGES ===\n")

# We only iterate up to 'n'
for i, row in tqdm(df.head(n).iterrows(), total=n):
    img_id = row["photo_id"]
    url = row["photo_image_url"]
    
    # Validation: Skip generic placeholder URLs or non-images if necessary
    if not isinstance(url, str) or len(url) < 5:
        continue

    if url.lower().endswith(".jpg"): #was getting issue with URls ending with jpg so now skipping them
        print(f"Skipping {url}")
        continue

    out_path = images_dir / f"{img_id}.jpg"

    if out_path.exists():
        continue

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        out_path.write_bytes(response.content)
    except Exception as e:
        # print(f"Failed: {img_id}") # Optional: Uncomment to see failures
        pass


# 3. INITIALIZE MODEL & DB
print("\n=== INITIALIZING MODEL & DB ===")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

db = lancedb.connect(uri="./lancedb_db")

# Define Schema
class ImageRecord(LanceModel):
    photo_id: str
    photo_image_url: str
    vector: Vector(512)

# Create Table (Overwrite ensures fresh start)
table = db.create_table("images", schema=ImageRecord, mode="overwrite")


# 4. EMBEDDING AND INGESTION
print("\n=== GENERATING EMBEDDINGS ===")

def load_image_safe(path: Path): # Load image with safety checks
    try:
        # Check size before fully loading
        with Image.open(path) as img:
            # Skip if image has more than 100 million pixels
            if (img.width * img.height) > 100000000: 
                print(f"Skipping too large image: {path.name}")
                return None
            return img.convert("RGB")
            
    except (UnidentifiedImageError, OSError, UserWarning):
        return None
    except Exception as e:
        # Catch the specific DecompressionBombError if it escalates
        if "DecompressionBomb" in str(e):
            print(f"Skipping massive image (DecompressionBomb): {path.name}")
            return None
        return None

target_ids = set(df.head(n)['photo_id'].astype(str))

# Only include paths that match these IDs
all_paths = sorted(images_dir.glob("*.jpg"))
image_paths = [p for p in all_paths if p.stem in target_ids]

print(f"Found {len(image_paths)} images to process out of {n} requested.")
for i in tqdm(range(0, len(image_paths), batch_size)):
    # Prepare Batch
    batch_paths = image_paths[i : i + batch_size]
    batch_images = []
    valid_paths = []
    
    for path in batch_paths:
        img = load_image_safe(path)
        if img:
            batch_images.append(img)
            valid_paths.append(path)
            
    if not batch_images:
        continue

    try:
        # Process and Embed
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs) # Get image embeddings
        
        # Normalize
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy()

        # Prepare Data for LanceDB
        batch_data = []
        for idx, path in enumerate(valid_paths):
            p_id = path.stem
            # Use .get() with default string to avoid crashing if ID is missing
            url = id_to_url.get(p_id, "unknown_url") 
            
            record = ImageRecord( 
                photo_id=p_id, 
                photo_image_url=url,
                vector=embeddings[idx] 
            )
            batch_data.append(record) #append metadata and vector together
            
        # Insert
        table.add(batch_data) #push vectors and metadata to LanceDB
        
    except Exception as e:
        print(f"Error processing batch: {e}")

print(f"\nSuccess! Indexed {len(table)} images into LanceDB.")


# The following part is just for verification and is not part of the main script. (optional to comment it out )

# 1. Connect to your existing database
db = lancedb.connect("./lancedb_db")

# 2. Open the table you created
try:
    table = db.open_table("images")
    
    # CHECK 1: How many images are inside?
    count = len(table)
    print(f"\n✅ Success! The table contains {count} images.\n")

    # CHECK 2: Show me the first 5 rows
    # We convert it to a Pandas DataFrame to make it readable
    df = table.search().limit(5).to_pandas()
    
    # Display columns
    print("--- Sample Data ---")
    print(df[["photo_id", "photo_image_url", "vector"]])

except Exception as e:
    print(f"❌ Could not open table: {e}")
    print("Make sure you ran the main script first and the folder './lancedb_db' exists.")