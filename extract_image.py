import cv2
import os
import uuid
from insightface.app import FaceAnalysis

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================


VIDEO_PATH = "/home/user/Desktop/video surveillence/Sample_videos/Export__Mahidharpura Nr Temple Thoba Sheri_Friday February 20 2026114459  a430615.avi"  # <-- PUT YOUR VIDEO PATH HERE
OUTPUT_FOLDER = "extreme image"

PROCESS_EVERY_N_FRAMES = 10  # Extract a face every 10 frames
CONFIDENCE_THRESHOLD = 0.35  # Ignore blurry/uncertain faces
MIN_FACE_SIZE = 15          # Ignore faces smaller than 50x50 pixels
PADDING_RATIO = 0.2          # Add 20% padding around the face for a better crop

# ==========================================
# 🚀 INITIALIZE AI (DETECTION ONLY)
# ==========================================
print("Loading high-res face detector on GPU...")
# We only need 'detection', skipping 'recognition' makes it incredibly fast
app = FaceAnalysis(name='antelopev2', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1920, 1920))

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created folder: {OUTPUT_FOLDER}")

# ==========================================
# 🎬 PROCESS VIDEO
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
        break  # End of video

    frame_count += 1

    # Skip frames to avoid saving thousands of identical photos
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue

    # Detect faces
    faces = app.get(frame)

    for face in faces:
        # 1. Filter by confidence
        if face.det_score < CONFIDENCE_THRESHOLD:
            continue

        # 2. Get bounding box coordinates
        x1, y1, x2, y2 = face.bbox.astype(int)

        # 3. Filter by size (ignore tiny faces in the distant background)
        width = x2 - x1
        height = y2 - y1
        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue

        # 4. Add Padding (Expands the box so it doesn't cut off hair/chin)
        pad_w = int(width * PADDING_RATIO)
        pad_h = int(height * PADDING_RATIO)
        
        # Ensure coordinates don't go outside the video frame
        img_h, img_w, _ = frame.shape
        px1 = max(0, x1 - pad_w)
        py1 = max(0, y1 - pad_h)
        px2 = min(img_w, x2 + pad_w)
        py2 = min(img_h, y2 + pad_h)

        # 5. Crop and Save
        face_crop = frame[py1:py2, px1:px2]
        
        if face_crop.size > 0:
            filename = f"face_f{frame_count}_{str(uuid.uuid4())[:6]}.jpg"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(filepath, face_crop)
            faces_saved += 1

    # Print a simple progress update
    if frame_count % 300 == 0:
        print(f"Processed {frame_count}/{total_frames} frames... Saved {faces_saved} faces so far.")

cap.release()
print(f"\n🎉 Extraction Complete! Saved {faces_saved} high-quality face images to '{OUTPUT_FOLDER}'.")