import cv2
import chromadb
import uuid
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime
import os
import sys

# ==========================================
# CONFIG
# ==========================================

VIDEO_PATH = "/home/user/Desktop/video surveillence/Sample_videos/Night videos/Export__Central Bus Depo-Exit_Thursday March 05 202651728  1adc061.avi"

PROCESS_EVERY_N_FRAMES = 30
FACE_MATCH_THRESHOLD = 0.50

os.environ["ORT_LOGGING_LEVEL"] = "3"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "live_video_db")
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "captured_faces")

# ==========================================
# IMAGE ENHANCEMENT
# ==========================================

def gamma_correction(image, gamma=1.4):

    invGamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


def apply_clahe(frame):

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def reduce_glare(frame):

    return cv2.convertScaleAbs(frame, alpha=1.1, beta=-20)


def is_blurry(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < 100


def enhance_night_frame(frame):

    frame = reduce_glare(frame)
    frame = gamma_correction(frame,1.4)
    frame = apply_clahe(frame)

    return frame


# ==========================================
# DATABASE
# ==========================================

print("Connecting to ChromaDB...")

try:

    db_client = chromadb.PersistentClient(path=DB_PATH)

    collection = db_client.get_or_create_collection(
        name="face_embeddings",
        metadata={"hnsw:space":"cosine"}
    )

    print("ChromaDB connected")

except Exception as e:

    print("Database Error:",e)
    sys.exit()


# ==========================================
# AI MODEL
# ==========================================

print("Loading InsightFace model...")

app = FaceAnalysis(
    name='antelopev2',
    providers=['CUDAExecutionProvider','CPUExecutionProvider']
)

app.prepare(ctx_id=0, det_size=(640,640))

print("Model loaded")


# ==========================================
# VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("Cannot open video")
    sys.exit()

frame_count = 0
processed_frames = 0
total_faces_detected = 0
faces_saved = 0

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

print("Processing started...")

# ==========================================
# MAIN LOOP
# ==========================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        processed_frames += 1

        print(f"\nProcessing frame #{frame_count}")

        # Night enhancement
        frame = enhance_night_frame(frame)

        # Resize
        frame = cv2.resize(frame, None, fx=0.9, fy=0.9)

        faces = app.get(frame)

        face_count = len(faces)

        total_faces_detected += face_count

        print(f"Faces detected in this frame: {face_count}")

        if faces:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for face in faces:

                if face.det_score < 0.45:
                    continue

                x1,y1,x2,y2 = face.bbox.astype(int)

                y1 = max(0,y1)
                y2 = min(frame.shape[0],y2)

                x1 = max(0,x1)
                x2 = min(frame.shape[1],x2)

                face_crop = frame[y1:y2,x1:x2]

                if face_crop.size == 0:
                    continue

                h,w = face_crop.shape[:2]

                if w < 20 or h < 20:
                    continue

                if is_blurry(face_crop):
                    continue

                embedding = face.embedding.tolist()

                person_id = None

                # ==========================================
                # SEARCH IN DATABASE
                # ==========================================

                if collection.count() > 0:

                    results = collection.query(
                        query_embeddings=[embedding],
                        n_results=1
                    )

                    if results['distances']:

                        distance = results['distances'][0][0]

                        if distance < FACE_MATCH_THRESHOLD:

                            person_id = results['metadatas'][0][0]["person_id"]

                            print("MATCH FOUND:",person_id)

                # ==========================================
                # CREATE NEW PERSON
                # ==========================================

                if not person_id:

                    person_id = "person_"+str(uuid.uuid4())[:8]

                    print("NEW PERSON:",person_id)

                person_folder = os.path.join(SAVE_FOLDER, person_id)

                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)

                img_id = str(uuid.uuid4())[:8]

                filename = f"{timestamp}_{img_id}.jpg"

                filepath = os.path.join(person_folder, filename)

                cv2.imwrite(filepath, face_crop)

                collection.add(

                    ids=[str(uuid.uuid4())],

                    embeddings=[embedding],

                    metadatas={
                        "person_id": person_id,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_path": filepath
                    }
                )

                faces_saved += 1

                print("Face saved:", filepath)

        print(f"Frames processed so far: {processed_frames}")
        print(f"Total faces detected: {total_faces_detected}")
        print(f"Faces saved: {faces_saved}")


except KeyboardInterrupt:

    print("Stopped by user")

finally:

    cap.release()

    print("\n========== FINAL STATS ==========")
    print("Total frames read:",frame_count)
    print("Frames processed:",processed_frames)
    print("Total faces detected:",total_faces_detected)
    print("Faces saved:",faces_saved)
    print("Processing finished")