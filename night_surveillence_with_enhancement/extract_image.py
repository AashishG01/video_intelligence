import cv2
import os
import uuid
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH             = "/home/user/Desktop/video surveillence/Sample_videos/Night videos/Export__Zampa Bazar-Towards Air India Office_Tuesday March 17 2026112634  f19c3cd.avi"
OUTPUT_FOLDER          = "testing image raw"
PROCESS_EVERY_N_FRAMES = 10
CONFIDENCE_THRESHOLD   = 0.35
MIN_FACE_SIZE          = 30
PADDING_RATIO          = 0.2

# ==========================================
# INITIALIZE AI (DETECTION ONLY)
# ==========================================

print("⏳ Loading face detector...")

app = FaceAnalysis(
    name='antelopev2',
    allowed_modules=['detection'],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

app.prepare(ctx_id=0, det_size=(1280, 1280))

# Mugshot override — catch small/angled faces
app.models['detection'].det_thresh = 0.10

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"✅ Created folder: {OUTPUT_FOLDER}")

# ==========================================
# PROCESS VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Error: Could not open video {VIDEO_PATH}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✅ Video loaded. Total frames: {total_frames}. Starting extraction...\n")

frame_count = 0
faces_saved = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue

    # Detect on raw frame — no enhancement
    faces = app.get(frame)

    for face in faces:

        # 1. Confidence filter
        if face.det_score < CONFIDENCE_THRESHOLD:
            continue

        # 2. Bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)

        # 3. Size filter
        width  = x2 - x1
        height = y2 - y1

        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue

        # 4. Add padding
        pad_w = int(width  * PADDING_RATIO)
        pad_h = int(height * PADDING_RATIO)

        img_h, img_w = frame.shape[:2]

        px1 = max(0,     x1 - pad_w)
        py1 = max(0,     y1 - pad_h)
        px2 = min(img_w, x2 + pad_w)
        py2 = min(img_h, y2 + pad_h)

        # 5. Crop from raw frame and save
        face_crop = frame[py1:py2, px1:px2]

        if face_crop.size > 0:
            filename = f"face_f{frame_count}_{str(uuid.uuid4())[:6]}.jpg"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(filepath, face_crop)
            faces_saved += 1

    if frame_count % 300 == 0:
        print(f"   Processed {frame_count}/{total_frames} frames... Saved {faces_saved} faces so far.")

cap.release()

print(f"\n✅ Extraction Complete! Saved {faces_saved} face images to '{OUTPUT_FOLDER}'.")