import streamlit as st
import chromadb
import pandas as pd
from PIL import Image
import os

st.set_page_config(layout="wide", page_title="ChromaDB Face Explorer")

st.title("👤 Face Detection Database Explorer")

# Connect to your DB
DB_PATH = "live_video_db"
if os.path.exists(DB_PATH):
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Get collection names
    collections = client.list_collections()
    col_names = [c.name for c in collections]
    
    if col_names:
        selected_col = st.sidebar.selectbox("Select Collection", col_names)
        collection = client.get_collection(selected_col)
        
        # Get data
        data = collection.get()
        
        if data['ids']:
            # Create a dataframe for the metadata
            df = pd.DataFrame(data['metadatas'])
            df['id'] = data['ids']
            
            st.write(f"Showing {len(df)} records")
            
            # Display entries with images
            for index, row in df.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        img_path = row.get('image_path', '')
                        if os.path.exists(img_path):
                            st.image(img_path, width=150)
                        else:
                            st.warning("No Image")
                            
                    with col2:
                        st.subheader(f"ID: {row['face_id']}")
                        st.text(f"Camera: {row['camera_id']} | Time: {row['timestamp']}")
                        st.code(f"BBox (YOLO): {row['yolo_bbox']}")
                        st.divider()
        else:
            st.info("Collection is empty.")
    else:
        st.error("No collections found in database.")
else:
    st.error(f"Database path not found: {DB_PATH}")