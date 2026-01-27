#!/usr/bin/env python3
"""Simple script to test LanceDB connection without requiring ML dependencies."""
import os
from dotenv import load_dotenv
import lancedb

load_dotenv()

DEFAULT_DB_URI = os.getenv("LANCEDB_URI", "db://segp-htni9f")
LANCEDB_API_KEY = os.getenv("LANCEDB_API_KEY", "sk_...")
LANCEDB_REGION = os.getenv("LANCEDB_REGION", "us-east-1")

def test_connection():
    """Test LanceDB connection and return True if successful."""
    print(f"[TEST] Testing LanceDB connection...")
    print(f"[TEST] URI: {DEFAULT_DB_URI}")
    print(f"[TEST] Region: {LANCEDB_REGION}")
    api_key_display = '*' * (len(LANCEDB_API_KEY) - 4) + LANCEDB_API_KEY[-4:] if len(LANCEDB_API_KEY) > 4 else '***'
    print(f"[TEST] API Key: {api_key_display}")
    print()
    
    try:
        print("[INFO] Attempting to connect...")
        db = lancedb.connect(
            uri=DEFAULT_DB_URI,
            api_key=LANCEDB_API_KEY,
            region=LANCEDB_REGION
        )
        print("[INFO] Connection established! Listing tables...")
        table_names = db.table_names()
        print(f"[SUCCESS] ✓ Connection successful!")
        print(f"[INFO] Found {len(table_names)} table(s)")
        if table_names:
            for table in table_names:
                print(f"  - {table}")
        else:
            print("[INFO] No tables found (this is expected if you haven't ingested data yet)")
        return True
    except Exception as e:
        print(f"[ERROR] ✗ Connection failed: {e}")
        print()
        print("Full error details:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)

