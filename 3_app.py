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

st.title("SEGP G20 Multimodal Search")
model, processor, table, device = load_system()

query = st.text_input("Search:", "Mountain")

# Added a slider to control the number of results
no_result = st.slider("Number of results to show:", min_value=1, max_value=50, value=9)

if query:
    # Convert text query to vector
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        q_vec = features.cpu().numpy()[0].tolist()
    
    # Search results from lancedb
    results = table.search(q_vec).limit(no_result).to_pandas()

    # Display results
    st.subheader(f"Found {len(results)} matches")
    
    if not results.empty:
        # Create a grid of 3 columns
        cols = st.columns(3)
        for idx, row in results.iterrows():
            # This math places images in the next available column
            with cols[idx % 3]:
                st.image(row["path"], use_container_width=True)
                st.caption(f"File: {row['filename']}")
    else:
        st.warning("No results found...")