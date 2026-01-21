import streamlit as st
import lancedb
import torch
from transformers import CLIPModel, CLIPProcessor

@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    db = lancedb.connect("./lancedb_data")
    table = db.open_table("my_images")
    return model, processor, table, device

st.title("Search Engine V2")
model, processor, table, device = load_system()

query = st.text_input("Search:", "a dog in the park")

if query:
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        q_vec = features.cpu().numpy()[0].tolist()
    
    results = table.search(q_vec).limit(3).to_pandas()
    
    cols = st.columns(3)
    for idx, row in results.iterrows():
        with cols[idx % 3]:
            st.image(row['path'], use_container_width=True)