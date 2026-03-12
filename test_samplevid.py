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
# Replace these with the paths to your local .avi files
VIDEO_PATH_1 = "/home/user/Desktop/video surveillence/Sample_videos/Export__Rly Station Parking Area-Entry_Wednesday February 18 202694311  aa4cf13.avi" 
VIDEO_PATH_2 = "/home/user/Desktop/video surveillence/Sample_videos/Export__Rly Station-Towards Amisha Hotel Right_Friday February 20 2026114208  608d3f6.avi" 
VIDEO_PATH_3 = "/home/user/Desktop/video surveillence/Sample_videos/Export__MajuraGate-Towards sagrampura_Wednesday February 18 202695109  1a65bd9.avi"
VIDEO_PATH_4 = "/home/user/Desktop/video surveillence/Sample_videos/Export__Mahidharpura Nr Temple Thoba Sheri_Friday February 20 2026114459  a430615.avi"

# For recorded video, we skip frames instead of using time.sleep()
# Assuming a 30 FPS video, setting this to 30 processes 1 frame per second of video time.
PROCESS_EVERY_N_FRAMES = 30 

# Match strictness
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
# VIDEO FILE STREAM CLASS (No Threading Needed)
# ==========================================
class VideoStream:
    def __init__(self, src, cam_id):
        self.cam_id = cam_id
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.frame_count = 0
        self.stopped = False
        
        if not self.stream.isOpened():
            print(f"⚠️  WARNING: Cannot open video file {src}.")
            self.stopped = True
        else:
            # Calculate total frames for progress tracking
            self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"✅ {cam_id} loaded successfully. ({self.total_frames} frames)")

    def read(self):
        if self.stopped:
            return False, None
            
        grabbed, frame = self.stream.read()
        if not grabbed:
            print(f"🎬 {self.cam_id} has reached the end of the video.")
            self.stopped = True
            return False, None
            
        self.frame_count += 1
        return True, frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# PROCESSING LOGIC
# ==========================================
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def process_gpu(vid_obj):
    if vid_obj.stopped:
        return False # Tells the main loop this video is done
        
    grabbed, frame = vid_obj.read()
    if not grabbed or frame is None:
        return False

    # Skip frames to speed up processing (Simulates the 1-second interval)
    if vid_obj.frame_count % PROCESS_EVERY_N_FRAMES != 0:
        return True # Continue without running the heavy AI

    faces = app.get(frame)
    if faces:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        for face in faces:
            if face.det_score < 0.6: continue
            
            x1, y1, x2, y2 = face.bbox.astype(int)
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            face_crop = frame[y1:y2, x1:x2]

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
                    
                    print(f"🔍 DEBUG [{vid_obj.cam_id} - Frame {vid_obj.frame_count}]: Closest face dist: {distance:.4f}")
                    
                    if distance < FACE_MATCH_THRESHOLD:
                        person_id = results['metadatas'][0][0].get("person_id")
                        print(f"✅ MATCH: Grouping with existing {person_id}")
                    else:
                        print(f"❌ NO MATCH: Distance higher than {FACE_MATCH_THRESHOLD}. Creating new ID.")
            
            # CREATE NEW PERSON IF NO MATCH
            if not person_id:
                person_id = f"person_{str(uuid.uuid4())[:8]}"
                print(f"🆕 NEW PERSON: Creating folder for {person_id}")

            person_folder = os.path.join(SAVE_FOLDER, person_id)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

            img_id = str(uuid.uuid4())[:8]
            file_name = f"{vid_obj.cam_id}_{timestamp_str}_{img_id}.jpg"
            file_path = os.path.join(person_folder, file_name)
            cv2.imwrite(file_path, face_crop)

            collection.add(
                ids=[str(uuid.uuid4())], 
                embeddings=[embedding],
                metadatas={
                    "person_id": person_id,
                    "cam": vid_obj.cam_id, 
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": file_path 
                }
            )

    return True

# ==========================================
# MAIN EXECUTION
# ==========================================
print("🚀 Starting Video Processing...")
vid1 = VideoStream(VIDEO_PATH_1, "Video_01")
vid2 = VideoStream(VIDEO_PATH_2, "Video_02")
vid3 = VideoStream(VIDEO_PATH_3, "Video_03")
vid4 = VideoStream(VIDEO_PATH_4, "Video_04")

print("🟢 System Running. Press Ctrl+C to exit.")

try:
    while True:
        # Process frames from all 4 videos
        active1 = process_gpu(vid1)
        active2 = process_gpu(vid2)
        active3 = process_gpu(vid3)
        active4 = process_gpu(vid4)
        
        # If all 4 videos are finished (stopped), exit the loop
        if not active1 and not active2 and not active3 and not active4:
            print("\n🏁 All videos have finished processing.")
            break

except KeyboardInterrupt:
    print("\n🛑 Process interrupted by user.gi")
finally:
    # Ensure all video files are cleanly closed to prevent memory leaks
    vid1.stop()
    vid2.stop()
    vid3.stop()
    vid4.stop()
    print("System Closed.")