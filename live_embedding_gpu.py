import cv2
import chromadb
import threading
import time
import uuid
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Replace with your actual RTSP Camera Links
RTSP_URL_1 = "rtsp://admin:admin@172.16.0.151:554/live.sdp" 
RTSP_URL_2 = "rtsp://admin:admin@172.16.0.152:554/live.sdp" 

# GPU is fast, so we can process more often (e.g., every 0.1 seconds)
PROCESS_INTERVAL = 0.1 

print("⏳ Initializing GPU System...")

db_client = chromadb.PersistentClient(path="live_video_db")
collection = db_client.get_or_create_collection(name="face_embeddings")

print("🧠 Attempting to load AI on NVIDIA GPU...")

RTSP_URL_1 = "rtsp://admin:admin@172.16.0.151:554/live.sdp" 
# 1. Force ONLY CUDA (Remove CPU from list)
# This asks InsightFace to please only use GPU.
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. STRICT VERIFICATION
# Even if we ask for CUDA, sometimes it fails silently. We must check what actually loaded.
try:
    # Check the provider of the detection model
    active_providers = app.models['detection'].session.get_providers()
    
    if 'CUDAExecutionProvider' not in active_providers:
        raise RuntimeError("CUDA Provider missing from active session.")
        
    print(f"✅ SUCCESS: AI running on {active_providers[0]}")

except Exception as e:
    print("\n" + "="*50)
    print("❌ CRITICAL ERROR: GPU NOT FOUND OR FAILED TO LOAD")
    print("="*50)
    print(f"Details: The system fell back to CPU or failed completely.")
    print("Reason: likely missing 'cudnn64_9.dll' or incompatible drivers.")
    print("Action: Stopping code to prevent slow CPU processing.")
    print("="*50 + "\n")
    sys.exit(1)  # <--- STOPS THE CODE HERE

print("✅ GPU System Ready. Connecting to Cameras...")

# ==========================================
# THREADED CAMERA CLASS (Non-Blocking)
# ==========================================
class CameraStream:
    def __init__(self, src, cam_id):
        self.cam_id = cam_id
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.last_process_time = time.time()

    def start(self):
        # Start the background thread
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# GPU PROCESSING FUNCTION
# ==========================================
def process_gpu(cam_obj):
    """
    Grabs the latest frame and sends it to the GPU for face extraction.
    """
    current_time = time.time()
    
    # Throttling to prevent processing the exact same frame twice
    if current_time - cam_obj.last_process_time < PROCESS_INTERVAL:
        return

    frame = cam_obj.read()
    if frame is None:
        return

    # 1. Run AI on GPU (Instant)
    faces = app.get(frame)

    if len(faces) > 0:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{cam_obj.cam_id}]  Detected {len(faces)} faces at {timestamp_str}")

        for face in faces:
            # 2. Extract Data
            embedding = face.embedding.tolist()
            img_id = str(uuid.uuid4())
            db_key = f"{cam_obj.cam_id}_{img_id}"

            # 3. Save to DB
            metadata = {
                "camera_id": cam_obj.cam_id,
                "timestamp": timestamp_str,
                "img_id": img_id,
                "confidence": float(face.det_score)
            }

            collection.add(
                ids=[db_key],
                embeddings=[embedding],
                metadatas=[metadata]
            )
    
    cam_obj.last_process_time = current_time

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
# Start Cameras
cam1 = CameraStream(RTSP_URL_1, "Camera_01").start()
cam2 = CameraStream(RTSP_URL_2, "Camera_02").start()

# Warmup time for cameras
time.sleep(2.0)

print("🚀  Indexing Started. Press Ctrl+C to stop.")

try:
    while True:
        # GPU is powerful enough to handle these sequentially with near-zero latency
        process_gpu(cam1)
        process_gpu(cam2)
        
        # Tiny sleep to prevent Python from eating 100% of a single CPU core
        time.sleep(0.005) 

except KeyboardInterrupt:
    print("\n🛑 Stopping GPU System...")
    cam1.stop()
    cam2.stop()
    print("System Closed.")