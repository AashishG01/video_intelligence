import cv2
import chromadb
import threading
import time
import uuid
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime
import sys
import os

# ==========================================
# CONFIGURATION
# ==========================================
RTSP_URL_1 = "rtsp://admin:admin@172.16.0.151:554/live.sdp" 
RTSP_URL_2 = "rtsp://admin:admin@172.16.0.152:554/live.sdp" 

# DEBUG CHANGE: Slowed down to 1 frame per second to make terminal readable
PROCESS_INTERVAL = 1.0 

# Our current threshold. We will adjust this based on the debug prints!
FACE_MATCH_THRESHOLD = 0.45 

os.environ["ORT_LOGGING_LEVEL"] = "3" 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "live_video_db")
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "captured_faces")

print("⏳ Step 1: Connecting to ChromaDB...")
try:
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(
        name="face_embeddings",
        metadata={"hnsw:space": "cosine"} 
    )
    print(f"✅ ChromaDB Connected at {DB_PATH}")
except Exception as e:
    print(f"❌ ChromaDB Error: {e}")
    sys.exit(1)

print("⏳ Step 2: Loading AI Models...")
app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1024, 1024))

active_providers = app.models['detection'].session.get_providers()
print(f"✅ AI System Ready. Active Provider: {active_providers[0]}")

# ==========================================
# THREADED CAMERA CLASS 
# ==========================================
class CameraStream:
    def __init__(self, src, cam_id):
        self.cam_id = cam_id
        self.src = src
        self.stopped = False
        self.frame = None
        self.last_process_time = time.time()
        
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        success, frame = self.stream.read()
        if not success:
            print(f"⚠️  WARNING: {cam_id} is unreachable.")
        else:
            self.frame = frame
            print(f"✅ {cam_id} is Streaming.")

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                time.sleep(1)
                self.stream = cv2.VideoCapture(self.src)
                continue
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# PROCESSING LOGIC
# ==========================================
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def process_gpu(cam_obj):
    if cam_obj.frame is None: return
    
    current_time = time.time()
    if current_time - cam_obj.last_process_time < PROCESS_INTERVAL:
        return

    faces = app.get(cam_obj.frame)
    if faces:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        for face in faces:
            if face.det_score < 0.6: continue
            
            x1, y1, x2, y2 = face.bbox.astype(int)
            y1, y2 = max(0, y1), min(cam_obj.frame.shape[0], y2)
            x1, x2 = max(0, x1), min(cam_obj.frame.shape[1], x2)
            face_crop = cam_obj.frame[y1:y2, x1:x2]

            if face_crop.size == 0: continue

            embedding = face.embedding.tolist()
            person_id = None

            # SEARCH DB FOR EXISTING PERSON
            if collection.count() > 0:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=1
                )
                
                if results['distances'] and len(results['distances'][0]) > 0:
                    distance = results['distances'][0][0]
                    
                    # ---------------------------------------------------------
                    # DEBUG PRINT: This tells us exactly what ChromaDB is seeing
                    # ---------------------------------------------------------
                    print(f"🔍 DEBUG [{cam_obj.cam_id}]: Closest face distance is {distance:.4f} (Threshold is {FACE_MATCH_THRESHOLD})")
                    
                    if distance < FACE_MATCH_THRESHOLD:
                        person_id = results['metadatas'][0][0].get("person_id")
                        print(f"✅ MATCH: Grouping with existing {person_id}")
                    else:
                        print(f"❌ NO MATCH: Distance {distance:.4f} is higher than {FACE_MATCH_THRESHOLD}. Creating new ID.")
            
            # CREATE NEW PERSON IF NO MATCH
            if not person_id:
                person_id = f"person_{str(uuid.uuid4())[:8]}"
                print(f"🆕 NEW PERSON: Creating folder for {person_id}")

            person_folder = os.path.join(SAVE_FOLDER, person_id)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

            img_id = str(uuid.uuid4())[:8]
            file_name = f"{cam_obj.cam_id}_{timestamp_str}_{img_id}.jpg"
            file_path = os.path.join(person_folder, file_name)
            cv2.imwrite(file_path, face_crop)

            collection.add(
                ids=[str(uuid.uuid4())], 
                embeddings=[embedding],
                metadatas={
                    "person_id": person_id,
                    "cam": cam_obj.cam_id, 
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": file_path 
                }
            )

    cam_obj.last_process_time = current_time

# ==========================================
# MAIN EXECUTION
# ==========================================
print("🚀 Starting Stream Threads...")
cam1 = CameraStream(RTSP_URL_1, "Cam_01").start()
cam2 = CameraStream(RTSP_URL_2, "Cam_02").start()

print("🟢 System Running. Press Ctrl+C to exit.")

try:
    while True:
        process_gpu(cam1)
        process_gpu(cam2)
        time.sleep(0.01) 
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    cam1.stop()
    cam2.stop()