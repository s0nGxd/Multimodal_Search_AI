import os
import numpy as np
import streamlit as st
import torch
from PIL import Image
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from lancedb import connect

load_dotenv()
DEFAULT_DB_URI = os.getenv("LANCEDB_URI", "./lancedb_db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "openai/clip-vit-base-patch32")

@torch.no_grad()
def encode_text(model, processor, text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    vec = model.get_text_features(**inputs)
    vec = vec / vec.norm(p=2, dim=-1, keepdim=True)
    return vec.cpu().numpy().astype("float32")[0]

st.set_page_config(page_title="SEGP Multimodal Search", layout="wide")
st.title("🔎 SEGP Multimodal Search (CLIP + LanceDB)")

db = connect(DEFAULT_DB_URI)
table_names = db.table_names()
if "images" not in table_names:
    st.warning("No 'images' table found. Please run `ingest.py` first.")
else:
    tbl = db.open_table("images")
    with st.sidebar:
        st.header("Settings")
        k = st.slider("Top-K", min_value=1, max_value=20, value=6)

    with st.spinner("Loading CLIP..."):
        model = CLIPModel.from_pretrained(EMBED_MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(EMBED_MODEL_NAME)
        model.eval()

    query = st.text_input("Describe the image you want to find:", "a dog catching a frisbee in a sunny park")
    if st.button("Search") or query:
        qvec = encode_text(model, processor, query)
        results = (
            tbl.search(qvec)
               .metric("cosine")
               .limit(k)
               .select(["path","text","vector"])
               .to_pandas()
        )
        if results.empty:
            st.info("No results yet — add images to ./data and run ingest.py")
        else:
            cols = st.columns(min(k, 4))
            for i, (_, row) in enumerate(results.iterrows()):
                col = cols[i % len(cols)]
                with col:
                    st.caption(f"{i+1}. {row['text']}")
                    try:
                        st.image(Image.open(row["path"]), use_column_width=True)
                    except Exception:
                        st.write(row["path"])
