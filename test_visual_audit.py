import os
import cv2
import chromadb
import numpy as np
from insightface.app import FaceAnalysis
import sys

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
TEST_FOLDER = "extreme image"
DB_PATH = "live_video_db"
AUDIT_FOLDER = "visual_audit_results_for_extreme"
MATCH_THRESHOLD = 0.55  # Let's test the 86% sweet spot

# Create output directories
os.makedirs(os.path.join(AUDIT_FOLDER, "matches"), exist_ok=True)
os.makedirs(os.path.join(AUDIT_FOLDER, "mismatches"), exist_ok=True)

# ==========================================
# 🚀 INITIALIZATION
# ==========================================
print(f"⏳ Connecting to ChromaDB at '{DB_PATH}'...")
try:
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_collection(name="face_embeddings")
    print(f"✅ Connected! DB has {collection.count()} records.")
except Exception as e:
    print(f"❌ DB Error: {e}")
    sys.exit(1)

print("⏳ Loading AntelopeV2 AI...")
app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
app.models['detection'].det_thresh = 0.10  # Mugshot override

print("✅ AI Ready. Starting Visual Audit...\n")

# ==========================================
# 🛠️ HELPER FUNCTION FOR IMAGE STITCHING
# ==========================================
def create_comparison_image(test_img, db_img, distance, person_id, is_match):
    """Resizes, stitches, and adds text to create a comparison card."""
    # Resize both to 256x256 for a clean side-by-side look
    test_resized = cv2.resize(test_img, (256, 256))
    
    if db_img is not None:
        db_resized = cv2.resize(db_img, (256, 256))
    else:
        # If DB image file is missing, create a black placeholder
        db_resized = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.putText(db_resized, "IMG MISSING", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Stitch them together horizontally (Left: Test, Right: DB Match)
    combined = cv2.hconcat([test_resized, db_resized])

    # Create a black bar at the bottom for text
    text_bar = np.zeros((60, 512, 3), dtype=np.uint8)
    
    # Text formatting
    color = (0, 255, 0) if is_match else (0, 0, 255) # Green for Match, Red for Mismatch
    status_text = f"MATCH" if is_match else f"MISMATCH"
    info_text = f"{status_text} | ID: {person_id} | Dist: {distance:.4f}"
    
    cv2.putText(text_bar, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(text_bar, "Test Image -->", (40, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(text_bar, "<-- DB Record", (350, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # Stack the images on top of the text bar
    final_img = cv2.vconcat([combined, text_bar])
    return final_img

# ==========================================
# 🧪 TESTING LOOP
# ==========================================
total_images = 0

print("="*60)
print(f"🔍 GENERATING VISUAL AUDIT (Threshold: {MATCH_THRESHOLD})")
print("="*60)

for filename in os.listdir(TEST_FOLDER):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    total_images += 1
    filepath = os.path.join(TEST_FOLDER, filename)
    
    test_img = cv2.imread(filepath)
    if test_img is None:
        continue

    faces = app.get(test_img)
    if not faces:
        print(f"[{total_images}] ⚠️ Skipping {filename} (No face detected even with override)")
        continue

    suspect_embedding = faces[0].embedding.tolist()

    results = collection.query(
        query_embeddings=[suspect_embedding],
        n_results=1
    )

    if results['distances'] and len(results['distances'][0]) > 0:
        distance = results['distances'][0][0]
        meta = results['metadatas'][0][0]
        person_id = meta.get("person_id", "Unknown")
        db_img_path = meta.get("image_path", "")
        
        is_match = distance < MATCH_THRESHOLD

        # Try to load the image from the database path
        db_img = cv2.imread(db_img_path) if os.path.exists(db_img_path) else None
        
        # Create the visual comparison card
        audit_img = create_comparison_image(test_img, db_img, distance, person_id, is_match)

        # Save it to the right folder
        subfolder = "matches" if is_match else "mismatches"
        save_name = f"dist_{distance:.4f}_{filename}"
        cv2.imwrite(os.path.join(AUDIT_FOLDER, subfolder, save_name), audit_img)
        
        print(f"[{total_images}] Saved audit card: {save_name} -> {subfolder.upper()}")

print("="*60)
print(f"✅ Audit Complete! Open the '{AUDIT_FOLDER}' folder to review the images.")
print("="*60)