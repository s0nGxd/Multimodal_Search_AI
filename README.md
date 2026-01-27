# SEGP Multimodal Search – Starter

This starter sets up a Python environment, connects to **LanceDB**, and gives you
scripts to **generate CLIP embeddings**, **ingest into LanceDB**, **create an index**, and **run a simple query**.

---

## 1) Create & activate a virtualenv

```bash
# Python 3.10+ recommended
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## 2) Install dependencies

> For most CPUs (no GPU):
```bash
pip install -r requirements.txt
```

> (Optional) If you need a specific CPU wheel for torch on older Macs/Windows, check:
https://pytorch.org/get-started/locally/

## 3) Configure environment (optional)
Create a `.env` (or export env vars) if you want to change defaults:
```
LANCEDB_URI=./lancedb_db
EMBED_MODEL=openai/clip-vit-base-patch32
```

## 4) Ingest images → LanceDB
Put some sample images into `./data/` (nested folders are okay). Then run:
```bash
python ingest.py --image_dir ./data --limit 200
```

## 5) Run a quick semantic search (text → images)
```bash
python search_demo.py --query "a dog catching a frisbee in a sunny park" --k 5
```

## 6) (Optional) Launch the Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

---

### Files
- `requirements.txt` – core libraries
- `ingest.py` – builds image embeddings with CLIP, writes to LanceDB, and creates an IVF-PQ index
- `search_demo.py` – encodes a text query and retrieves top matches
- `app/streamlit_app.py` – minimal UI for searching
- `notebooks/quick_test.ipynb` – empty notebook you can use to experiment

**Default LanceDB location**: `./lancedb_db` (created on first run)

