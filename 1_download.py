import pandas as pd
import glob
import requests
import os
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
TARGET_IMAGES = 200     # Number of images to download for the prototype
OUTPUT_DIR = "images"   # Folder to save images
DATA_PATH = "data/raw/unsplash_dataset/" # Update this to point to where your csv000 files are

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load the Dataset (Adapted from your snippet) ---
print("Loading dataset...")

# We specifically look for the 'photos' file. 
# Your file is likely named 'photos.csv000', so we search for 'photos.csv*'
files = glob.glob(os.path.join(DATA_PATH, "photos.csv*"))

if not files:
    print(f"Error: No photos file found in {DATA_PATH}. Check your path!")
    exit()

# Read the file (using sep='\t' as per the official snippet)
documents = []
for filename in files:
    print(f"Reading {filename}...")
    df = pd.read_csv(filename, sep='\t', header=0)
    documents.append(df)

# Combine if there are multiple parts
photos_df = pd.concat(documents, axis=0, ignore_index=True)

print(f"Successfully loaded metadata for {len(photos_df)} images.")

# --- 2. Download Logic ---
def download_image(row):
    try:
        # Get ID and URL from the Unsplash dataframe
        photo_id = row['photo_id']
        photo_url = row['photo_image_url']
        
        # Skip if already exists
        if os.path.exists(f"{OUTPUT_DIR}/{photo_id}.jpg"):
            return True

        # Download
        response = requests.get(photo_url, timeout=5)
        if response.status_code == 200:
            with open(f"{OUTPUT_DIR}/{photo_id}.jpg", "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        return False
    return False

# --- 3. Execute Download ---
print(f"Downloading first {TARGET_IMAGES} images...")

# Select the first N rows
subset = photos_df.head(TARGET_IMAGES)

# Run download in parallel (faster)
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(lambda x: download_image(x[1]), subset.iterrows()))

success_count = sum(results)
print(f"Download complete. {success_count} images saved to '{OUTPUT_DIR}/'.")
print("You are now ready for Step 2 (Ingestion)!")