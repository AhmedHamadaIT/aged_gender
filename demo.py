import streamlit as st
import requests
import os

# FORCE ABSOLUTE PATH
GALLERY_DIR = "/content/ml-server/gallery"
API_URL = "http://localhost:8000/reid/search"

st.set_page_config(layout="wide")
st.title("ReID Dashboard")

# Debug info: This will show on your dashboard to help us troubleshoot
if not os.path.exists(GALLERY_DIR):
    st.error(f"🚨 Path NOT found: {GALLERY_DIR}")
    st.write("Current directory contents:", os.listdir('/content/ml-server/'))
else:
    st.success(f"✅ Path found! Total images: {len(os.listdir(GALLERY_DIR))}")

# --- Sidebar ---
st.sidebar.header("Search")
uploaded_file = st.sidebar.file_uploader("Search by Image", type=['jpg', 'jpeg', 'png'])

# FIXED INDENTATION HERE
if uploaded_file:
    st.sidebar.image(uploaded_file, width=150)
    if st.sidebar.button("Run Search"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            r = requests.post(API_URL, files=files, data={"top_k": 10})
            res_json = r.json()
            
            if res_json.get("status") == "success" and res_json.get("results"):
                st.subheader("Search Results")
                results = res_json["results"]
                
                # Create columns for the results
                res_cols = st.columns(len(results))
                
                for i, item in enumerate(results):
                    score = item.get("score", 0)
                    meta = item.get("metadata", {})
                    
                    # Extract the filename from "./gallery/image.jpg"
                    filename = os.path.basename(meta.get("image_path", ""))
                    
                    # Build the absolute path for Colab
                    abs_path = os.path.join(GALLERY_DIR, filename)
                    
                    with res_cols[i % len(res_cols)]:
                        if os.path.exists(abs_path):
                            st.image(
                                abs_path, 
                                caption=f"ID: {meta.get('track_id')} | Score: {score:.4f}", 
                                width='stretch' # Fixes the Streamlit warnings
                            )
                        else:
                            st.error(f"Missing: {filename}")
            else:
                st.warning("No matches found or API error.")
                
        except Exception as e:
            st.error(f"API Error: {e}")

st.divider()

# --- Gallery ---
st.subheader("Gallery View")
if os.path.exists(GALLERY_DIR):
    images = [f for f in os.listdir(GALLERY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    cols = st.columns(6)
    for idx, img_name in enumerate(images):
        # FIXED WARNING HERE: changed use_container_width to width='stretch'
        cols[idx % 6].image(os.path.join(GALLERY_DIR, img_name), caption=img_name, width='stretch')
