import cv2
import chromadb
import uuid
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime
import sys
import os
import time
import psutil
from pynvml import *

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "/home/user/Desktop/video surveillence/Sample_videos/Export__Rly Station Parking Area-Entry_Wednesday February 18 202694311  aa4cf13.avi"

PROCESS_EVERY_N_FRAMES = 30
FACE_MATCH_THRESHOLD = 0.45

os.environ["ORT_LOGGING_LEVEL"] = "3"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "live_video_db")
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "captured_faces")

# ==========================================
# GPU INIT
# ==========================================

gpu_available = False

try:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_available = True
except:
    print("⚠️ GPU monitoring not available")

# ==========================================
# DATABASE
# ==========================================

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

# ==========================================
# AI MODEL
# ==========================================

print("⏳ Step 2: Loading AI Models...")

app = FaceAnalysis(
    name='antelopev2',
    providers=['CUDAExecutionProvider','CPUExecutionProvider']
)

app.prepare(ctx_id=0, det_size=(1024,1024))

print("✅ AI System Ready")

# ==========================================
# VIDEO STREAM
# ==========================================

class VideoStream:

    def __init__(self, src):

        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.frame_count = 0
        self.stopped = False

        if not self.stream.isOpened():
            print("⚠️ Cannot open video")
            self.stopped = True
        else:
            self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)

            print(f"✅ Video loaded ({self.total_frames} frames)")
            print(f"📹 Camera FPS: {self.fps}")

    def read(self):

        if self.stopped:
            return False, None

        grabbed, frame = self.stream.read()

        if not grabbed:
            self.stopped = True
            return False, None

        self.frame_count += 1
        return True, frame

    def stop(self):
        self.stream.release()

# ==========================================
# SETUP
# ==========================================

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

video = VideoStream(VIDEO_PATH)

print("🚀 Processing Started")

# ==========================================
# METRIC VARIABLES
# ==========================================

start_time = time.time()

processed_frames = 0
faces_detected = 0
faces_saved = 0

cpu_usage_log = []
ram_usage_log = []

gpu_usage_log = []
gpu_mem_log = []
gpu_power_log = []

# ==========================================
# MAIN LOOP
# ==========================================

try:

    while True:

        grabbed, frame = video.read()

        if not grabbed:
            break

        if video.frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        processed_frames += 1

        # CPU + RAM
        cpu_usage_log.append(psutil.cpu_percent())
        ram_usage_log.append(psutil.virtual_memory().percent)

        # GPU metrics
        if gpu_available:

            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            power = nvmlDeviceGetPowerUsage(handle) / 1000

            gpu_usage_log.append(util.gpu)
            gpu_mem_log.append(mem.used / (1024**3))
            gpu_power_log.append(power)

        faces = app.get(frame)

        faces_detected += len(faces)

        if faces:

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            for face in faces:

                if face.det_score < 0.6:
                    continue

                x1,y1,x2,y2 = face.bbox.astype(int)

                y1 = max(0,y1)
                y2 = min(frame.shape[0],y2)

                x1 = max(0,x1)
                x2 = min(frame.shape[1],x2)

                face_crop = frame[y1:y2,x1:x2]

                if face_crop.size == 0:
                    continue

                embedding = face.embedding.tolist()
                person_id = None

                if collection.count() > 0:

                    results = collection.query(
                        query_embeddings=[embedding],
                        n_results=1
                    )

                    if results['distances']:

                        distance = results['distances'][0][0]

                        if distance < FACE_MATCH_THRESHOLD:
                            person_id = results['metadatas'][0][0]["person_id"]

                if not person_id:

                    person_id = f"person_{str(uuid.uuid4())[:8]}"

                person_folder = os.path.join(SAVE_FOLDER, person_id)

                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)

                img_id = str(uuid.uuid4())[:8]

                filename = f"{timestamp_str}_{img_id}.jpg"
                filepath = os.path.join(person_folder, filename)

                cv2.imwrite(filepath, face_crop)

                collection.add(

                    ids=[str(uuid.uuid4())],

                    embeddings=[embedding],

                    metadatas={
                        "person_id":person_id,
                        "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_path":filepath
                    }
                )

                faces_saved += 1

except KeyboardInterrupt:

    print("Interrupted")

finally:

    video.stop()

# ==========================================
# FINAL ANALYSIS
# ==========================================

end_time = time.time()
runtime = end_time - start_time

processing_fps = processed_frames / runtime
frame_latency = runtime / processed_frames if processed_frames else 0

avg_cpu = np.mean(cpu_usage_log)
avg_ram = np.mean(ram_usage_log)

print("\n==============================")
print("📊 BENCHMARK REPORT")
print("==============================")

print(f"Runtime: {runtime:.2f} sec")
print(f"Total Frames Read: {video.frame_count}")
print(f"Frames Processed: {processed_frames}")

print(f"Processing FPS: {processing_fps:.2f}")
print(f"Frame Latency: {frame_latency:.4f} sec")

print(f"Faces Detected: {faces_detected}")
print(f"Faces Saved: {faces_saved}")

print("\n🖥 CPU / RAM")
print(f"Average CPU Usage: {avg_cpu:.2f}%")
print(f"Average RAM Usage: {avg_ram:.2f}%")

if gpu_available:

    avg_gpu = np.mean(gpu_usage_log)
    avg_vram = np.mean(gpu_mem_log)
    avg_power = np.mean(gpu_power_log)

    print("\n🎮 GPU Metrics")
    print(f"Average GPU Usage: {avg_gpu:.2f}%")
    print(f"Average GPU Memory: {avg_vram:.2f} GB")
    print(f"Average GPU Power: {avg_power:.2f} W")

    # Estimate camera capacity
    camera_fps = video.fps
    required_processing_fps = camera_fps / PROCESS_EVERY_N_FRAMES

    cameras_per_gpu = processing_fps / required_processing_fps

    print("\n📹 ESTIMATED CAPACITY")
    print(f"Camera FPS: {camera_fps}")
    print(f"Frames processed per camera: {required_processing_fps:.2f}")
    print(f"Estimated cameras per GPU: {int(cameras_per_gpu)}")

print("\n✅ System Closed")