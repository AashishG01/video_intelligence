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
import psutil
from pynvml import *

# ==========================================
# CONFIGURATION
# ==========================================
RTSP_URL_1 = "rtsp://admin:admin@172.16.0.151:554/live.sdp"
RTSP_URL_2 = "rtsp://admin:admin@172.16.0.152:554/live.sdp"
RTSP_URL_3 = "rtsp://admin:123456@172.16.0.161:554/live.sdp"
RTSP_URL_4 = "rtsp://admin:Admin@123@172.16.0.162:554/live.sdp"

PROCESS_INTERVAL     = 1.0
FACE_MATCH_THRESHOLD = 0.45
RUN_DURATION_SECONDS = 120  # 2 minutes

os.environ["ORT_LOGGING_LEVEL"] = "3"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(SCRIPT_DIR, "Database")
DB_PATH      = os.path.join(DATABASE_DIR, "live_video_db")
SAVE_FOLDER  = os.path.join(DATABASE_DIR, "captured_faces")

os.makedirs(DB_PATH,     exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==========================================
# GPU INIT
# ==========================================
gpu_available = False
try:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_available = True
    print("✅ GPU Monitoring Ready.")
except:
    print("⚠️  GPU monitoring not available.")

# ==========================================
# DATABASE
# ==========================================
print("⏳ Step 1: Connecting to ChromaDB...")
try:
    db_client  = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection(
        name="face_embeddings",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✅ ChromaDB Connected at {DB_PATH}")
except Exception as e:
    print(f"❌ ChromaDB Error: {e}")
    sys.exit(1)

# ==========================================
# AI MODEL
# ==========================================
print("⏳ Step 2: Loading AI Models...")
app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1024, 1024))

active_providers = app.models['detection'].session.get_providers()
print(f"✅ AI System Ready. Active Provider: {active_providers[0]}")


# ==========================================
# THREADED CAMERA CLASS (FIXED)
# ==========================================
class CameraStream:
    def __init__(self, src, cam_id):
        self.cam_id    = cam_id
        self.src       = src
        self.stopped   = False
        self.frame     = None
        self.connected = False
        self.last_process_time = time.time()
        self.thread    = None

        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        success, frame = self.stream.read()
        if not success:
            print(f"❌ {cam_id} is UNREACHABLE — {src}")
            self.connected = False
        else:
            self.frame     = frame
            self.connected = True
            print(f"✅ {cam_id} is Streaming.")

    def start(self):
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                time.sleep(1)
                # Ensure the old stream is released before reconnecting to prevent memory leaks
                if self.stream.isOpened():
                    self.stream.release()
                self.stream = cv2.VideoCapture(self.src)
                continue
            self.frame = frame
        
        # ✅ FIX: Release the stream safely INSIDE the worker thread after the loop ends
        if self.stream and self.stream.isOpened():
            self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        # ✅ FIX: Wait for the thread to recognize the stop flag and release the stream
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

# ==========================================
# BENCHMARK VARIABLES
# ==========================================
start_time = time.time()

frames_processed = 0
faces_detected   = 0
faces_matched    = 0
faces_new        = 0
faces_saved      = 0

cpu_log         = []
ram_log         = []
gpu_log         = []
gpu_mem_log     = []
gpu_pow_log     = []
inference_times = []

# ==========================================
# HELPER
# ==========================================
def safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0

# ==========================================
# PROCESSING LOGIC
# ==========================================
def process_gpu(cam_obj):
    global frames_processed, faces_detected, faces_matched, faces_new, faces_saved

    if cam_obj.frame is None:
        return

    current_time = time.time()
    if current_time - cam_obj.last_process_time < PROCESS_INTERVAL:
        return

    # ── System metrics ──────────────────────
    cpu_log.append(psutil.cpu_percent())
    ram_log.append(psutil.virtual_memory().percent)

    if gpu_available:
        util  = nvmlDeviceGetUtilizationRates(handle)
        mem   = nvmlDeviceGetMemoryInfo(handle)
        power = nvmlDeviceGetPowerUsage(handle) / 1000
        gpu_log.append(util.gpu)
        gpu_mem_log.append(mem.used / (1024 ** 3))
        gpu_pow_log.append(power)

    frames_processed += 1

    # ── AI Inference ────────────────────────
    t_start = time.perf_counter()
    faces   = app.get(cam_obj.frame)
    t_end   = time.perf_counter()
    inference_times.append(t_end - t_start)

    if faces:
        faces_detected += len(faces)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        for face in faces:
            if face.det_score < 0.6:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)
            y1, y2 = max(0, y1), min(cam_obj.frame.shape[0], y2)
            x1, x2 = max(0, x1), min(cam_obj.frame.shape[1], x2)
            face_crop = cam_obj.frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            embedding = face.embedding.tolist()
            person_id = None

            if collection.count() > 0:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=1
                )

                if results['distances'] and len(results['distances'][0]) > 0:
                    distance = results['distances'][0][0]
                    print(f"🔍 [{cam_obj.cam_id}]: dist={distance:.4f}")

                    if distance < FACE_MATCH_THRESHOLD:
                        person_id = results['metadatas'][0][0].get("person_id")
                        faces_matched += 1
                        print(f"✅ MATCH: {person_id}")
                    else:
                        print(f"❌ NO MATCH")

            if not person_id:
                person_id = f"person_{str(uuid.uuid4())[:8]}"
                faces_new += 1
                print(f"🆕 NEW PERSON: {person_id}")

            person_folder = os.path.join(SAVE_FOLDER, person_id)
            os.makedirs(person_folder, exist_ok=True)

            img_id    = str(uuid.uuid4())[:8]
            file_name = f"{cam_obj.cam_id}_{timestamp_str}_{img_id}.jpg"
            file_path = os.path.join(person_folder, file_name)
            cv2.imwrite(file_path, face_crop)
            faces_saved += 1

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                metadatas={
                    "person_id":  person_id,
                    "cam":        cam_obj.cam_id,
                    "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": file_path
                }
            )

    cam_obj.last_process_time = current_time

# ==========================================
# MAIN EXECUTION
# ==========================================
print("🚀 Starting Stream Threads...")

cam1 = CameraStream(RTSP_URL_1, "Lab_Cam_01").start()
cam2 = CameraStream(RTSP_URL_2, "Lab_Cam_02").start()
cam3 = CameraStream(RTSP_URL_3, "Lab_Cam_03").start()
cam4 = CameraStream(RTSP_URL_4, "Department_gate").start()

# ── Camera Status Summary ──────────────────
print("\n" + "=" * 40)
print("📷 CAMERA STATUS SUMMARY")
print("=" * 40)
for cam in [cam1, cam2, cam3, cam4]:
    status = "🟢 ONLINE " if cam.connected else "🔴 OFFLINE"
    print(f"  {status} — {cam.cam_id}")
print("=" * 40 + "\n")

print(f"🟢 System Running for {RUN_DURATION_SECONDS} seconds...")

try:
    while True:
        elapsed = time.time() - start_time

        if elapsed >= RUN_DURATION_SECONDS:
            print(f"\n⏰ {RUN_DURATION_SECONDS} seconds completed. Stopping...")
            break

        process_gpu(cam1)
        process_gpu(cam2)
        process_gpu(cam3)
        process_gpu(cam4)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n🛑 Interrupted")

finally:
    cam1.stop()
    cam2.stop()
    cam3.stop()
    cam4.stop()

    if gpu_available:
        nvmlShutdown()

   
    print("✅ All resources released.")

# ==========================================
# FINAL BENCHMARK REPORT
# ==========================================
end_time = time.time()
runtime  = end_time - start_time

avg_infer_ms = safe_mean(inference_times) * 1000
avg_cpu      = safe_mean(cpu_log)
avg_ram      = safe_mean(ram_log)

online_cameras  = sum(1 for c in [cam1, cam2, cam3, cam4] if c.connected)
offline_cameras = 4 - online_cameras

print("\n" + "=" * 60)
print("📊 BENCHMARK REPORT — LIVE RTSP (4 Cameras)")
print("=" * 60)
print(f"Runtime                  : {runtime:.2f} sec")
print(f"Cameras Online           : {online_cameras} / 4")
print(f"Cameras Offline          : {offline_cameras} / 4")
print(f"Frames Processed (total) : {frames_processed}")
print(f"Avg Processing Rate      : {frames_processed / runtime:.2f} frames/sec")
print("-" * 60)
print(f"Faces Detected           : {faces_detected}")
print(f"Faces Matched (existing) : {faces_matched}")
print(f"New Persons Created      : {faces_new}")
print(f"Faces Saved              : {faces_saved}")
print("-" * 60)
print(f"Avg AI Inference Time    : {avg_infer_ms:.2f} ms / frame")
print("-" * 60)
print(f"Avg CPU Usage            : {avg_cpu:.2f}%")
print(f"Avg RAM Usage            : {avg_ram:.2f}%")

if gpu_available:
    avg_gpu   = safe_mean(gpu_log)
    avg_vram  = safe_mean(gpu_mem_log)
    avg_power = safe_mean(gpu_pow_log)

    print(f"Avg GPU Usage            : {avg_gpu:.2f}%")
    print(f"Avg GPU Memory           : {avg_vram:.2f} GB")
    print(f"Avg GPU Power            : {avg_power:.2f} W")

    required_fps_per_cam = 1 / PROCESS_INTERVAL
    max_cameras          = (1000 / avg_infer_ms) / required_fps_per_cam if avg_infer_ms > 0 else 0

    print("-" * 60)
    print(f"Processing Interval      : {PROCESS_INTERVAL} sec")
    print(f"Estimated Max Cameras    : {int(max_cameras)}")

print("=" * 60)
print("✅ Benchmark Complete")

# ✅ Force cleanup — segfault prevent karta hai
import gc
del collection
del db_client
gc.collect()

# ✅ Thoda wait karo DB ko settle hone do
time.sleep(2)