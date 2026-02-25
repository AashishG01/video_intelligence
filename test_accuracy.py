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
MATCH_THRESHOLD = 0.55  # Must match your API's threshold

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

# *** THE MUGSHOT OVERRIDE ***
# This forces the AI to process pre-cropped faces that your backend would normally reject.
app.models['detection'].det_thresh = 0.10  

print("✅ AI Ready. Starting 6-Step Pipeline Test...\n")

# ==========================================
# 🧪 TESTING LOOP
# ==========================================
total_images = 0
match_count = 0
mismatch_count = 0

print("="*60)
print("🔍 RUNNING ACCURACY PIPELINE")
print("="*60)

for filename in os.listdir(TEST_FOLDER):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    total_images += 1
    filepath = os.path.join(TEST_FOLDER, filename)
    
    # ---------------------------------------------------------
    # STEP 1: Image Ingestion & Decoding
    # ---------------------------------------------------------
    img = cv2.imread(filepath)
    if img is None:
        continue

    # ---------------------------------------------------------
    # STEP 2: Biometric Vector Extraction
    # ---------------------------------------------------------
    faces = app.get(img)
    if not faces:
        # With the override above, this should almost never happen now.
        print(f"[{total_images}] ❌ FAILED AT STEP 2: Still couldn't extract vector for {filename}")
        mismatch_count += 1
        continue

    suspect_embedding = faces[0].embedding.tolist()

    # ---------------------------------------------------------
    # STEP 3: Metadata Filter Construction
    # ---------------------------------------------------------
    # In this test, we are simulating a global search (no camera/time filters)
    where_filter = None 

    # ---------------------------------------------------------
    # STEP 4: The Vector Database Query (ChromaDB)
    # ---------------------------------------------------------
    results = collection.query(
        query_embeddings=[suspect_embedding],
        n_results=1, # We only care about the absolute best match for this test
        where=where_filter
    )

    # ---------------------------------------------------------
    # STEP 5 & 6: Threshold Enforcement & Result Formatting
    # ---------------------------------------------------------
    if results['distances'] and len(results['distances'][0]) > 0:
        distance = results['distances'][0][0]
        person_id = results['metadatas'][0][0].get("person_id", "Unknown")
        
        # Enforce the strict < 0.50 threshold
        if distance < MATCH_THRESHOLD:
            match_count += 1
            confidence = (1.0 - distance) * 100
            print(f"[{total_images}] ✅ MATCH: '{filename}' -> {person_id} (Conf: {confidence:.1f}%)")
        else:
            mismatch_count += 1
            print(f"[{total_images}] ❌ MISMATCH: Nearest was {person_id}, but distance ({distance:.4f}) breached threshold.")
    else:
        mismatch_count += 1
        print(f"[{total_images}] ❌ MISMATCH: ChromaDB returned zero records.")

# ==========================================
# 📊 FINAL SCORECARD
# ==========================================
accuracy_percent = (match_count / total_images * 100) if total_images > 0 else 0

print("\n" + "="*60)
print("📊 FINAL PIPELINE ACCURACY REPORT")
print("="*60)
print(f"Total Test Images Scanned: {total_images}")
print("-" * 60)
print(f"✅ Successful Matches:      {match_count}")
print(f"❌ Failed / Mismatches:     {mismatch_count}")
print("-" * 60)
print(f"🎯 System Accuracy Rate:    {accuracy_percent:.2f}%")
print("="*60)