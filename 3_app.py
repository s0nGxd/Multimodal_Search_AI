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

# --- 3. Search Logic ---
st.subheader("Search Settings")
col1, col2 = st.columns(2)
with col1:
    # Existing slider: How many to fetch
    limit = st.slider("Max images to show", 1, 50, 9)
with col2:
    # NEW SLIDER: The Strictness Filter
    # Lower number = Stricter (Must be a very good match)
    # Higher number = Looser (Show me anything remotely close)
    threshold = st.slider("Strictness (Threshold)", 0.0, 1.5, 0.80, step=0.05)

if query:
    # 1. Vectorize
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy()[0].tolist()

    # 2. Search (Fetch more than we need, then filter)
    results = table.search(query_vector).limit(limit).to_pandas()

    # 3. FILTER BAD MATCHES
    # We keep only rows where '_distance' is LOWER than our threshold
    filtered_results = results[results['_distance'] < threshold]

    # 4. Display
    st.subheader(f"Matches found: {len(filtered_results)}")

    if not filtered_results.empty:
        cols = st.columns(3)
        for idx, row in filtered_results.iterrows():
            with cols[idx % 3]:
                st.image(row["path"], use_container_width=True)
                # Show the score so you can learn what "Good" looks like
                st.caption(f"Dist: {row['_distance']:.4f}")
    else:
        # This is the message you wanted!
        st.error(f"No images matched your description strictly enough (Threshold: {threshold}).")
        st.info("Try lowering the Strictness slider to see 'best guess' results.")