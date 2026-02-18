import cv2
import time
import uuid
import os
from datetime import datetime

# Import from our modular files
from config import RTSP_URL_1, RTSP_URL_2, PROCESS_INTERVAL, FACE_MATCH_THRESHOLD, SAVE_FOLDER
from database import VectorDB
from face_model import FaceEngine
from camera import CameraStream

# Ensure the main save directory exists
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Initialize our custom modules
db = VectorDB()
engine = FaceEngine()

def process_gpu(cam_obj):
    if cam_obj.frame is None: return
    
    current_time = time.time()
    if current_time - cam_obj.last_process_time < PROCESS_INTERVAL:
        return

    # Extract faces using the AI engine
    faces = engine.extract_faces(cam_obj.frame)
    
    if faces:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        for face in faces:
            # 1. Strict confidence check
            if face.det_score < 0.6: continue
            width = x2 - x1
            height = y2 - y1
            if width < 60 or height < 60:
                continue
            # 2. Crop the face (No yaw filter, let all angles pass)
            x1, y1, x2, y2 = face.bbox.astype(int)
            y1, y2 = max(0, y1), min(cam_obj.frame.shape[0], y2)
            x1, x2 = max(0, x1), min(cam_obj.frame.shape[1], x2)
            face_crop = cam_obj.frame[y1:y2, x1:x2]

            if face_crop.size == 0: continue
            blur_score = cv2.Laplacian(face_crop, cv2.CV_64F).var()
            if blur_score < 80.0:  # Adjust this number (50-100) based on your lighting
                continue
            if face.pose is not None:
                pitch, yaw, roll = face.pose
                if abs(yaw) > 60.0: # Ignore extreme side profiles
                    continue
            embedding = face.embedding.tolist()
            
            # 3. Database Search (Uses the new Top-3 breadcrumb logic)
            person_id, distance = db.search(embedding, FACE_MATCH_THRESHOLD)

            # 4. Handle Routing
            if person_id:
                print(f"✅ MATCH [{cam_obj.cam_id}]: Grouping with {person_id} (Dist: {distance:.4f})")
            else:
                dist_str = f"{distance:.4f}" if distance is not None else "Empty DB"
                print(f"❌ NO MATCH [{cam_obj.cam_id}]: Dist {dist_str}. Creating new ID.")
                person_id = f"person_{str(uuid.uuid4())[:8]}"
                print(f"🆕 NEW PERSON: Creating folder for {person_id}")

            # 5. Create specific person folder if needed
            person_folder = os.path.join(SAVE_FOLDER, person_id)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

            # 6. Save image to disk
            img_id = str(uuid.uuid4())[:8]
            file_name = f"{cam_obj.cam_id}_{timestamp_str}_{img_id}.jpg"
            file_path = os.path.join(person_folder, file_name)
            cv2.imwrite(file_path, face_crop)

            # 7. Save metadata and embedding to ChromaDB
            metadata = {
                "person_id": person_id,
                "cam": cam_obj.cam_id, 
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": file_path 
            }
            db.add_record(str(uuid.uuid4()), embedding, metadata)

    cam_obj.last_process_time = current_time

def main():
    print("🚀 Starting Stream Threads...")
    cam1 = CameraStream(RTSP_URL_1, "Cam_01").start()
    cam2 = CameraStream(RTSP_URL_2, "Cam_02").start()

    print("🟢 System Running. Press Ctrl+C to exit.")

    try:
        while True:
            process_gpu(cam1)
            process_gpu(cam2)
            
            # Tiny sleep prevents the while loop from maxing out 100% of a CPU core
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        cam1.stop()
        cam2.stop()

if __name__ == "__main__":
    main()