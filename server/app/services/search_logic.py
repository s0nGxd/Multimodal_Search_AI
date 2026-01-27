import os
import argparse
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import lancedb

load_dotenv()
DEFAULT_DB_URI = os.getenv("LANCEDB_URI", "db://segp-htni9f")
LANCEDB_API_KEY = os.getenv("LANCEDB_API_KEY", "sk_...")
LANCEDB_REGION = os.getenv("LANCEDB_REGION", "us-east-1")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "openai/clip-vit-base-patch32")

def test_connection():
    """Test LanceDB connection and return True if successful."""
    print(f"[TEST] Testing LanceDB connection...")
    print(f"[TEST] URI: {DEFAULT_DB_URI}")
    print(f"[TEST] Region: {LANCEDB_REGION}")
    print(f"[TEST] API Key: {'*' * (len(LANCEDB_API_KEY) - 4) + LANCEDB_API_KEY[-4:] if len(LANCEDB_API_KEY) > 4 else '***'}")
    
    try:
        db = lancedb.connect(
            uri=DEFAULT_DB_URI,
            api_key=LANCEDB_API_KEY,
            region=LANCEDB_REGION
        )
        table_names = db.table_names()
        print(f"[SUCCESS] Connection successful!")
        print(f"[INFO] Found {len(table_names)} table(s): {table_names}")
        return True
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@torch.no_grad()
def encode_text(model: CLIPModel, processor: CLIPProcessor, text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype("float32")[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=False, help="Natural language image query")
    parser.add_argument("--table", type=str, default="images")
    parser.add_argument("--k", type=int, default=5, help="Top-K")
    parser.add_argument("--test-connection", action="store_true", help="Test LanceDB connection and exit")
    args = parser.parse_args()
    
    if args.test_connection:
        success = test_connection()
        exit(0 if success else 1)
    
    if not args.query:
        parser.error("--query is required unless --test-connection is used")

    print(f"[INFO] Connecting to LanceDB at: {DEFAULT_DB_URI}")
    try:
        db = lancedb.connect(
            uri=DEFAULT_DB_URI,
            api_key=LANCEDB_API_KEY,
            region=LANCEDB_REGION
        )
        # Test connection by listing tables
        table_names = db.table_names()
        print(f"[SUCCESS] Connected to LanceDB! Found {len(table_names)} table(s): {table_names}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to LanceDB: {e}")
        return
    
    try:
        tbl = db.open_table(args.table)
        print(f"[INFO] Opened table '{args.table}'")
    except Exception as e:
        print(f"[ERROR] Failed to open table '{args.table}': {e}")
        print(f"[INFO] Available tables: {table_names}")
        return

    print(f"[INFO] Loading CLIP model: {EMBED_MODEL_NAME}")
    model = CLIPModel.from_pretrained(EMBED_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(EMBED_MODEL_NAME)
    model.eval()

    query_vec = encode_text(model, processor, args.query)

    print(f"[INFO] Searching top-{args.k} for: \"{args.query}\"")
    results = (
        tbl.search(query_vec)
           .metric("cosine")
           .limit(args.k)
           .select(["path", "text", "vector"])
           .to_pandas()
    )

    if results.empty:
        print("No results found. Did you ingest images?")
        return

    for i, row in results.iterrows():
        score = 1 - float(np.dot(row["vector"], query_vec))  # cosine distance → smaller is better
        print(f"[{i+1}] {row['path']}  | approx cosine distance: {score:.4f}  | text: {row['text']}")

if __name__ == "__main__":
    main()
