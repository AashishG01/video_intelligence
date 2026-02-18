import os
import cv2
import numpy as np
import chromadb
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis
from pydantic import BaseModel
from typing import List, Optional

# ==========================================
# SYSTEM SETUP
# ==========================================
# ==========================================
# SYSTEM SETUP
# ==========================================
# This gets the folder the API is in (e.g., /Desktop/video surveillence/backend)
API_DIR = os.path.dirname(os.path.abspath(__file__))

# This goes UP one level to the main project folder
PROJECT_DIR = os.path.dirname(API_DIR)

# Now we point to the exact location of the database and images!
DB_PATH = os.path.join(PROJECT_DIR, "live_video_db")
SAVE_FOLDER = os.path.join(PROJECT_DIR, "captured_faces")
app = FastAPI(
    title="Investigation API",
    description="Backend for real-time surveillance dashboard",
    version="1.0.0"
)

# Allow frontend dashboards to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the cropped face images so the frontend can display them!
app.mount("/images", StaticFiles(directory=SAVE_FOLDER), name="images")

# ==========================================
# AI & DATABASE INITIALIZATION
# ==========================================
print("⏳ Connecting to Surveillance Database...")
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(
    name="face_embeddings",
    metadata={"hnsw:space": "cosine"}
)

print("⏳ Loading Analysis AI...")
face_app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640)) # Smaller det_size is fine for uploaded mugshots
print("✅ API System Ready.")

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/api/system/stats")
async def get_system_stats():
    """Returns global dashboard statistics."""
    total_records = collection.count()
    
    if total_records == 0:
        return {"total_faces_captured": 0, "unique_suspects": 0, "active_cameras": 0}

    # Fetch all metadata to calculate stats
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data["metadatas"]

    unique_persons = set(m["person_id"] for m in metadatas if "person_id" in m)
    active_cameras = set(m["cam"] for m in metadatas if "cam" in m)
    
    # Sort to find the oldest record (Database Start Time)
    times = sorted([m["time"] for m in metadatas if "time" in m])
    db_start_time = times[0] if times else "Unknown"

    return {
        "status": "ONLINE",
        "total_faces_captured": total_records,
        "unique_suspects": len(unique_persons),
        "active_cameras": len(active_cameras),
        "camera_ids": list(active_cameras),
        "system_start_time": db_start_time
    }

@app.post("/api/investigate/search_by_image")
async def search_by_image(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Query(None, description="Filter by specific camera"),
    start_time: Optional[str] = Query(None, description="Format: YYYY-MM-DD HH:MM:SS"),
    end_time: Optional[str] = Query(None, description="Format: YYYY-MM-DD HH:MM:SS"),
    threshold: float = Query(0.50, description="Match strictness. Lower = Stricter")
):
    """Upload a suspect's photo to find all matching sightings."""
    # 1. Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Extract Face Embedding
    faces = face_app.get(img)
    if not faces:
        raise HTTPException(status_code=404, detail="No face detected in uploaded image.")
    
    # Assuming the most prominent face in the uploaded photo
    suspect_embedding = faces[0].embedding.tolist()

    # 3. Build Metadata Filters (Sci-Fi filtering!)
    where_filter = {}
    and_clauses = []

    if camera_id:
        and_clauses.append({"cam": camera_id})
    if start_time:
        and_clauses.append({"time": {"$gte": start_time}})
    if end_time:
        and_clauses.append({"time": {"$lte": end_time}})

    if len(and_clauses) == 1:
        where_filter = and_clauses[0]
    elif len(and_clauses) > 1:
        where_filter = {"$and": and_clauses}

    # 4. Query Database
    results = collection.query(
        query_embeddings=[suspect_embedding],
        n_results=20, # Get top 20 sightings
        where=where_filter if where_filter else None
    )

    if not results['distances'] or len(results['distances'][0]) == 0:
        return {"message": "No matches found.", "sightings": []}

    sightings = []
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    for i in range(len(distances)):
        if distances[i] < threshold:
            meta = metadatas[i]
            
            # Convert local file path to API URL for the dashboard
            filename = os.path.basename(meta["image_path"])
            image_url = f"/images/{meta['person_id']}/{filename}"

            sightings.append({
                "person_id": meta["person_id"],
                "camera": meta["cam"],
                "timestamp": meta["time"],
                "match_score": round(1.0 - distances[i], 4), # Convert distance to a 0-1 confidence score
                "image_url": image_url
            })

    # Sort sightings by most recent first
    sightings.sort(key=lambda x: x["timestamp"], reverse=True)

    return {
        "suspect_found": len(sightings) > 0,
        "total_sightings": len(sightings),
        "sightings": sightings
    }

@app.get("/api/investigate/person/{person_id}")
async def get_person_dossier(person_id: str):
    """Retrieve the entire tracking history of a specific person ID."""
    results = collection.get(
        where={"person_id": person_id},
        include=["metadatas"]
    )

    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail="Person ID not found in database.")

    sightings = []
    for meta in results["metadatas"]:
        filename = os.path.basename(meta["image_path"])
        image_url = f"/images/{meta['person_id']}/{filename}"
        
        sightings.append({
            "camera": meta["cam"],
            "timestamp": meta["time"],
            "image_url": image_url
        })

    # Sort chronological
    sightings.sort(key=lambda x: x["timestamp"])

    return {
        "person_id": person_id,
        "total_sightings": len(sightings),
        "first_seen": sightings[0]["timestamp"],
        "last_seen": sightings[-1]["timestamp"],
        "locations": list(set(s["camera"] for s in sightings)),
        "timeline": sightings
    }

#  uvicorn api:app --host 0.0.0.0 --port 8000 --reload