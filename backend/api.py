import os
import cv2
import threading
import numpy as np
import chromadb
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from insightface.app import FaceAnalysis
from typing import Optional

# ==========================================
# SYSTEM SETUP
# ==========================================
API_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(API_DIR)
DB_PATH     = os.path.join(PROJECT_DIR, "Database", "live_video_db")
SAVE_FOLDER = os.path.join(PROJECT_DIR, "Database", "captured_faces")

os.makedirs(DB_PATH,     exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==========================================
# RTSP CAMERA CONFIG
# ==========================================
RTSP_URLS = {
    "cam1": "rtsp://admin:admin@172.16.0.151:554/live.sdp",
    "cam2": "rtsp://admin:admin@172.16.0.152:554/live.sdp",
    "cam3": "rtsp://admin:123456@172.16.0.161:554/live.sdp",
    "cam4": "rtsp://admin:Admin@123@172.16.0.162:554/live.sdp",
}

# Shared frame buffer — background threads write here
camera_frames = {}
camera_locks  = {}

# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(
    title="C.O.R.E. Surveillance API",
    description="Backend for real-time surveillance dashboard",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=SAVE_FOLDER), name="images")

# ==========================================
# DATABASE
# ==========================================
print("⏳ Connecting to Surveillance Database...")
db_client  = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(
    name="face_embeddings",
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ ChromaDB Connected. Records: {collection.count()}")

# ==========================================
# AI MODEL
# ==========================================
print("⏳ Loading InsightFace AntelopeV2 (Detection + Recognition)...")
face_app = FaceAnalysis(
    name='antelopev2',
    # No allowed_modules restriction — loads BOTH detection AND recognition
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Lower internal threshold to catch night/partial/angled faces
face_app.models['detection'].det_thresh = 0.10

print("✅ InsightFace Ready — Detection + Embedding extraction active.")

# ==========================================
# CAMERA STREAM FUNCTIONS
# ==========================================
def capture_camera(cam_id, rtsp_url):
    """Background thread — continuously reads RTSP frames"""
    camera_locks[cam_id] = threading.Lock()

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    print(f"📷 Camera thread started: {cam_id}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  {cam_id} disconnected. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            continue

        # Resize for bandwidth saving
        frame = cv2.resize(frame, (640, 360))

        with camera_locks[cam_id]:
            camera_frames[cam_id] = frame


def generate_mjpeg(cam_id):
    """MJPEG frame generator for StreamingResponse"""
    import time
    while True:
        if cam_id not in camera_frames:
            time.sleep(0.1)
            continue

        with camera_locks[cam_id]:
            frame = camera_frames[cam_id].copy()

        ret, buffer = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )


# ==========================================
# STARTUP — Start camera threads
# ==========================================
@app.on_event("startup")
async def start_camera_threads():
    for cam_id, url in RTSP_URLS.items():
        t = threading.Thread(
            target=capture_camera,
            args=(cam_id, url),
            daemon=True
        )
        t.start()
        print(f"🟢 Stream thread started: {cam_id}")


# ==========================================
# STREAM ENDPOINTS
# ==========================================
@app.get("/api/stream/{cam_id}")
async def video_stream(cam_id: str):
    """MJPEG stream endpoint for each camera"""
    if cam_id not in RTSP_URLS:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found.")

    return StreamingResponse(
        generate_mjpeg(cam_id),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


@app.get("/api/stream/status")
async def stream_status():
    """Returns which cameras are currently streaming"""
    return {
        cam_id: cam_id in camera_frames
        for cam_id in RTSP_URLS
    }


# ==========================================
# SYSTEM STATS ENDPOINT
# ==========================================
@app.get("/api/system/stats")
async def get_system_stats():
    """Returns global dashboard statistics."""
    total_records = collection.count()

    if total_records == 0:
        return {
            "status":               "ONLINE",
            "total_faces_captured": 0,
            "unique_suspects":      0,
            "active_cameras":       0,
            "camera_ids":           [],
            "system_start_time":    "Unknown"
        }

    all_data  = collection.get(include=["metadatas"])
    metadatas = all_data["metadatas"]

    unique_persons = set(m["person_id"]          for m in metadatas if "person_id" in m)
    active_cameras = set(m.get("cam", "unknown") for m in metadatas)
    times          = sorted([m["time"]           for m in metadatas if "time" in m])
    db_start_time  = times[0] if times else "Unknown"

    return {
        "status":               "ONLINE",
        "total_faces_captured": total_records,
        "unique_suspects":      len(unique_persons),
        "active_cameras":       len(active_cameras),
        "camera_ids":           list(active_cameras),
        "system_start_time":    db_start_time
    }


# ==========================================
# IMAGE SEARCH ENDPOINT
# ==========================================
@app.post("/api/investigate/search_by_image")
async def search_by_image(
    file:       UploadFile    = File(...),
    camera_id:  Optional[str] = Query(None, description="Filter by specific camera"),
    start_time: Optional[str] = Query(None, description="Format: YYYY-MM-DD HH:MM:SS"),
    end_time:   Optional[str] = Query(None, description="Format: YYYY-MM-DD HH:MM:SS"),
    threshold:  float         = Query(0.50, description="Match strictness. Lower = Stricter")
):
    """Upload a suspect photo to find all matching sightings."""

    # 1. Read uploaded image
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Extract face embedding using full InsightFace pipeline
    faces = face_app.get(img)
    if not faces:
        raise HTTPException(status_code=404, detail="No face detected in uploaded image.")

    # Use embedding from recognition module (512-d vector)
    suspect_embedding = faces[0].embedding.tolist()

    # 3. Build metadata filter
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
    else:
        where_filter = None

    # 4. Query database
    results = collection.query(
        query_embeddings=[suspect_embedding],
        n_results=20,
        where=where_filter
    )

    if not results['distances'] or len(results['distances'][0]) == 0:
        return {"message": "No matches found.", "sightings": []}

    sightings = []
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    for i in range(len(distances)):
        if distances[i] < threshold:
            meta      = metadatas[i]
            filename  = os.path.basename(meta["image_path"])
            image_url = f"/images/{meta['person_id']}/{filename}"

            sightings.append({
                "person_id":   meta["person_id"],
                "camera":      meta.get("cam", "unknown"),
                "timestamp":   meta["time"],
                "match_score": round(1.0 - distances[i], 4),
                "image_url":   image_url
            })

    sightings.sort(key=lambda x: x["timestamp"], reverse=True)

    return {
        "suspect_found":   len(sightings) > 0,
        "total_sightings": len(sightings),
        "sightings":       sightings
    }


# ==========================================
# PERSON DOSSIER ENDPOINT
# ==========================================
@app.get("/api/investigate/person/{person_id}")
async def get_person_dossier(person_id: str):
    """Retrieve full tracking history of a specific person."""
    results = collection.get(
        where={"person_id": person_id},
        include=["metadatas"]
    )

    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail="Person ID not found in database.")

    sightings = []
    for meta in results["metadatas"]:
        filename  = os.path.basename(meta["image_path"])
        image_url = f"/images/{meta['person_id']}/{filename}"

        sightings.append({
            "camera":    meta.get("cam", "unknown"),
            "timestamp": meta["time"],
            "image_url": image_url
        })

    sightings.sort(key=lambda x: x["timestamp"])

    return {
        "person_id":       person_id,
        "total_sightings": len(sightings),
        "first_seen":      sightings[0]["timestamp"],
        "last_seen":       sightings[-1]["timestamp"],
        "locations":       list(set(s["camera"] for s in sightings)),
        "timeline":        sightings
    }


# ==========================================
# RUN
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# ==========================================