import cv2
import chromadb
import uuid
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime
import sys
import os

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "/home/user/Desktop/video surveillence/Sample_videos/Night videos/Export__Rly Station-Towards Amisha Hotel General_Thursday March 05 202651946  ee084f7.avi"
CAM_ID     = "Railway_Station_Amisha_Cam_night"

# Zampa_Bazar_Cam 
# Gopi_Talav_Cam
# Mahidharpura_Cam
# MajuraGate_Cam
# Railway_Station_Amisha_Cam
# Railway_Station_Parking_Cam
# Prakas_Bakery_Cam
# Railway_Station_Amisha_Cam_night

# /home/user/Desktop/video surveillence/Sample_videos/Night videos/Export__Rly Station-Towards Amisha Hotel General_Thursday March 05 202651946  ee084f7.avi
# /home/user/Desktop/video surveillence/Sample_videos/Night videos/Export__Prakas Bakery - Towards Janta Dairy_Tuesday March 17 2026113215  e008f0b.avi
# /home/user/Desktop/video surveillence/Sample_videos/Night videos/Export__Zampa Bazar-Towards Air India Office_Tuesday March 17 2026112634  f19c3cd.avi

PROCESS_EVERY_N_FRAMES = 30
FACE_MATCH_THRESHOLD   = 0.50

HIGH_ANGLE_CAMERA = True

if HIGH_ANGLE_CAMERA:
    DET_SCORE_THRESHOLD = 0.35
    MIN_FACE_SIZE       = 50
    UPSCALE_FACTOR      = 1.5
    DET_SIZE            = (1024, 1024)
else:
    DET_SCORE_THRESHOLD = 0.55
    MIN_FACE_SIZE       = 60
    UPSCALE_FACTOR      = 1.0
    DET_SIZE            = (1024, 1024)

os.environ["ORT_LOGGING_LEVEL"] = "3"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(SCRIPT_DIR, "Database")
DB_PATH      = os.path.join(DATABASE_DIR, "live_video_db")
SAVE_FOLDER  = os.path.join(DATABASE_DIR, "captured_faces")

os.makedirs(DB_PATH,     exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==========================================
# NIGHT ENHANCEMENT FUNCTIONS
# ==========================================

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(frame):
    return cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)


def enhance_night_frame(frame):
    frame = denoise(frame)
    frame = gamma_correction(frame, gamma=1.5)
    frame = apply_clahe(frame)
    if UPSCALE_FACTOR != 1.0:
        frame = cv2.resize(frame, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                           interpolation=cv2.INTER_LANCZOS4)
    return frame

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

app = FaceAnalysis(
    name='antelopev2',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=DET_SIZE)

if hasattr(app.models.get('detection', None), 'det_thresh'):
    app.models['detection'].det_thresh = 0.10

active_providers = app.models['detection'].session.get_providers()
print(f"✅ AI System Ready. Active Provider: {active_providers[0]}")
print(f"📷 Camera mode : {'High-angle (street)' if HIGH_ANGLE_CAMERA else 'Face-level'}")
print(f"   det_score   : {DET_SCORE_THRESHOLD}  |  min face size: {MIN_FACE_SIZE}px  |  upscale: {UPSCALE_FACTOR}x")

# ==========================================
# VIDEO STREAM CLASS
# ==========================================

class VideoStream:

    def __init__(self, src):
        self.src         = src
        self.stream      = cv2.VideoCapture(src)
        self.frame_count = 0
        self.stopped     = False

        if not self.stream.isOpened():
            print(f"⚠️ Cannot open video file {src}")
            self.stopped = True
        else:
            self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps          = self.stream.get(cv2.CAP_PROP_FPS)
            print(f"✅ Video loaded successfully ({self.total_frames} frames @ {self.fps:.1f} fps)")

    def read(self):
        if self.stopped:
            return False, None
        grabbed, frame = self.stream.read()
        if not grabbed:
            print("🎬 Video finished.")
            self.stopped = True
            return False, None
        self.frame_count += 1
        return True, frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# SETUP
# ==========================================

video = VideoStream(VIDEO_PATH)
print("🚀 Processing Started")

# ==========================================
# MAIN LOOP
# ==========================================

try:

    while True:

        grabbed, frame = video.read()
        if not grabbed or frame is None:
            break

        if video.frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        enhanced = enhance_night_frame(frame)
        faces    = app.get(enhanced)

        if not faces:
            continue

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        for face in faces:

            if face.det_score < DET_SCORE_THRESHOLD:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)
            y1 = max(0, y1)
            y2 = min(enhanced.shape[0], y2)
            x1 = max(0, x1)
            x2 = min(enhanced.shape[1], x2)

            face_crop = enhanced[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            h, w = face_crop.shape[:2]
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            embedding = face.embedding.tolist()
            person_id = None

            # ── Search database ────────────────────
            if collection.count() > 0:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=1
                )

                if results['distances'] and len(results['distances'][0]) > 0:
                    distance = results['distances'][0][0]
                    print(f"🔍 DEBUG [{CAM_ID} - Frame {video.frame_count}]: dist={distance:.4f}")

                    if distance < FACE_MATCH_THRESHOLD:
                        person_id = results['metadatas'][0][0]["person_id"]
                        print(f"✅ MATCH with {person_id}")
                    else:
                        print("❌ NO MATCH")

            # ── New person ─────────────────────────
            if not person_id:
                person_id = f"person_{str(uuid.uuid4())[:8]}"
                print(f"🆕 NEW PERSON {person_id}")

            # ── Save face image — HAMESHA ──────────
            person_folder = os.path.join(SAVE_FOLDER, person_id)
            os.makedirs(person_folder, exist_ok=True)

            filename = f"{CAM_ID}_{timestamp_str}_{str(uuid.uuid4())[:8]}.jpg"
            filepath = os.path.join(person_folder, filename)
            cv2.imwrite(filepath, face_crop)

            # ── DB mein HAMESHA add karo ───────────
            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                metadatas={
                    "person_id":  person_id,
                    "cam":        CAM_ID,
                    "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": filepath
                }
            )

except KeyboardInterrupt:
    print("\n🛑 Interrupted")

finally:
    video.stop()
    print("✅ System Closed")