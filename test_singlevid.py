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

VIDEO_PATH = "/home/user/Desktop/video surveillence/Sample_videos/Export__Rly Station Parking Area-Entry_Wednesday February 18 202694311  aa4cf13.avi"
CAM_ID     = "Railway_Station_Parking_Cam"  # ✅ Camera ID add kiya

# Zampa_Bazar_Cam 
# Gopi_Talav_Cam
# Mahidharpura_Cam
# MajuraGate_Cam
# Railway_Station_Amisha_Cam
# Railway_Station_Parking_Cam

# VIDEO_PATH_1 = "/home/user/Desktop/video surveillence/Sample_videos/Export__Rly Station Parking Area-Entry_Wednesday February 18 202694311  aa4cf13.avi" 
# VIDEO_PATH_2 = "/home/user/Desktop/video surveillence/Sample_videos/Export__Rly Station-Towards Amisha Hotel Right_Friday February 20 2026114208  608d3f6.avi" 
# VIDEO_PATH_3 = "/home/user/Desktop/video surveillence/Sample_videos/Export__MajuraGate-Towards sagrampura_Wednesday February 18 202695109  1a65bd9.avi"
# VIDEO_PATH_4 = "/home/user/Desktop/video surveillence/Sample_videos/Export__Mahidharpura Nr Temple Thoba Sheri_Friday February 20 2026114459  a430615.avi"
#/home/user/Desktop/video surveillence/Sample_videos/Export__Gopi Talav-Towards Gopi Talav Gate_Saturday March 14 2026120440  6942298.avi
#/home/user/Desktop/video surveillence/Sample_videos/Export__Zampa Bazar-Towards Air India Office_Saturday March 14 2026120139  4ece73f.avi

PROCESS_EVERY_N_FRAMES = 30
FACE_MATCH_THRESHOLD   = 0.45

os.environ["ORT_LOGGING_LEVEL"] = "3"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(SCRIPT_DIR, "Database")
DB_PATH      = os.path.join(DATABASE_DIR, "live_video_db")
SAVE_FOLDER  = os.path.join(DATABASE_DIR, "captured_faces")

os.makedirs(DB_PATH,     exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

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
app.prepare(ctx_id=0, det_size=(1024, 1024))

active_providers = app.models['detection'].session.get_providers()
print(f"✅ AI System Ready. Active Provider: {active_providers[0]}")

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
            print(f"✅ Video loaded successfully ({self.total_frames} frames)")

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

        faces = app.get(frame)
        if not faces:
            continue

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        for face in faces:

            if face.det_score < 0.6:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)
            y1 = max(0, y1)
            y2 = min(frame.shape[0], y2)
            x1 = max(0, x1)
            x2 = min(frame.shape[1], x2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            embedding = face.embedding.tolist()
            person_id = None
            is_new    = False

            # ── Search in DB ───────────────────────
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
                is_new    = True
                print(f"🆕 NEW PERSON {person_id}")

            # ── Save face image ────────────────────
            person_folder = os.path.join(SAVE_FOLDER, person_id)
            os.makedirs(person_folder, exist_ok=True)

            filename = f"{CAM_ID}_{timestamp_str}_{str(uuid.uuid4())[:8]}.jpg"
            filepath = os.path.join(person_folder, filename)
            cv2.imwrite(filepath, face_crop)

            # ── Add to DB only if NEW person ───────
            if is_new:
                collection.add(
                    ids=[str(uuid.uuid4())],
                    embeddings=[embedding],
                    metadatas={
                        "person_id":  person_id,
                        "cam":        CAM_ID,       # ✅ cam field add kiya
                        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_path": filepath
                    }
                )

except KeyboardInterrupt:
    print("\n🛑 Interrupted")

finally:
    video.stop()
    print("✅ System Closed")