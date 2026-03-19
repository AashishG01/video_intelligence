import cv2
import numpy as np
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "/home/user/Desktop/video surveillence/Sample_videos/Night videos/Export__Rly Station-Towards Amisha Hotel General_Thursday March 05 202651946  ee084f7.avi"

# Which frame number to grab for testing
# Change this to test different moments in the video
TEST_FRAME_NUMBER = 300

UPSCALE_FACTOR = 1.5

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "enhancement_debug")

# ==========================================
# ENHANCEMENT FUNCTIONS
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


def upscale(frame):
    return cv2.resize(frame, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                      interpolation=cv2.INTER_LANCZOS4)

# ==========================================
# GRAB TEST FRAME
# ==========================================

print(f"📂 Opening video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video. Check VIDEO_PATH.")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✅ Video opened. Total frames: {total_frames}")
print(f"🎯 Grabbing frame #{TEST_FRAME_NUMBER}...")

cap.set(cv2.CAP_PROP_POS_FRAMES, TEST_FRAME_NUMBER)
ret, original = cap.read()
cap.release()

if not ret:
    print("❌ Could not read frame. Try a different TEST_FRAME_NUMBER.")
    sys.exit(1)

print(f"✅ Frame grabbed. Size: {original.shape[1]}x{original.shape[0]}")

# ==========================================
# RUN EACH PHASE
# ==========================================

print("\n🔧 Running enhancement phases...")

# Phase 1 — original
phase1 = original.copy()
print("   Phase 1: Original ✓")

# Phase 2 — denoise
phase2 = denoise(phase1)
print("   Phase 2: Denoise ✓")

# Phase 3 — gamma correction
phase3 = gamma_correction(phase2, gamma=1.5)
print("   Phase 3: Gamma correction ✓")

# Phase 4 — CLAHE
phase4 = apply_clahe(phase3)
print("   Phase 4: CLAHE ✓")

# Phase 5 — upscale
phase5 = upscale(phase4)
print("   Phase 5: Upscale ✓")

# ==========================================
# ADD LABELS TO EACH PHASE IMAGE
# ==========================================

def add_label(image, text, sub=""):
    out = image.copy()
    h, w = out.shape[:2]

    # dark bar at top
    cv2.rectangle(out, (0, 0), (w, 52), (20, 20, 20), -1)

    # main label
    cv2.putText(out, text, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # sub label
    if sub:
        cv2.putText(out, sub, (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return out


p1 = add_label(phase1, "Phase 1: Original",   "No processing applied")
p2 = add_label(phase2, "Phase 2: Denoise",    "Bilateral filter — removes grain")
p3 = add_label(phase3, "Phase 3: Gamma",      "Gamma 1.5 — lifts dark shadows")
p4 = add_label(phase4, "Phase 4: CLAHE",      "Local contrast equalisation")
p5 = add_label(phase5, "Phase 5: Upscale",    f"Lanczos {UPSCALE_FACTOR}x — larger faces")

# ==========================================
# SAVE INDIVIDUAL PHASE IMAGES
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

cv2.imwrite(os.path.join(OUTPUT_DIR, "phase1_original.jpg"),  p1)
cv2.imwrite(os.path.join(OUTPUT_DIR, "phase2_denoise.jpg"),   p2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "phase3_gamma.jpg"),     p3)
cv2.imwrite(os.path.join(OUTPUT_DIR, "phase4_clahe.jpg"),     p4)
cv2.imwrite(os.path.join(OUTPUT_DIR, "phase5_upscale.jpg"),   p5)

print(f"\n✅ Individual phase images saved to: {OUTPUT_DIR}")

# ==========================================
# BUILD SIDE-BY-SIDE COMPARISON GRID
# ==========================================

print("🖼️  Building comparison grid...")

# Resize all to same height for grid (use original size, not upscaled)
TARGET_H = 360
TARGET_W = int(original.shape[1] * (TARGET_H / original.shape[0]))

def resize_for_grid(img):
    return cv2.resize(img, (TARGET_W, TARGET_H))

g1 = resize_for_grid(p1)
g2 = resize_for_grid(p2)
g3 = resize_for_grid(p3)
g4 = resize_for_grid(p4)
g5 = resize_for_grid(p5)

# Row 1: Phase 1, 2, 3
row1 = np.hstack([g1, g2, g3])

# Row 2: Phase 4, Phase 5, blank placeholder
blank = np.zeros_like(g1)
cv2.putText(blank, "Enhancement", (30, TARGET_H // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
cv2.putText(blank, "Complete", (50, TARGET_H // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

row2 = np.hstack([g4, g5, blank])

# Divider line between rows
divider = np.full((4, row1.shape[1], 3), 60, dtype=np.uint8)

grid = np.vstack([row1, divider, row2])

grid_path = os.path.join(OUTPUT_DIR, "all_phases_grid.jpg")
cv2.imwrite(grid_path, grid)

print(f"✅ Full comparison grid saved: {grid_path}")

# ==========================================
# DIFF IMAGE — shows what changed overall
# ==========================================

print("📊 Generating diff image (Original vs Final)...")

# Resize final to match original for diff
final_resized = cv2.resize(phase4, (original.shape[1], original.shape[0]))

diff = cv2.absdiff(original, final_resized)
diff_boosted = cv2.convertScaleAbs(diff, alpha=3.0)  # boost visibility

diff_labeled = add_label(diff_boosted,
                          "Diff: Original vs Final",
                          "Bright areas = pixels changed by enhancement")

cv2.imwrite(os.path.join(OUTPUT_DIR, "diff_original_vs_final.jpg"), diff_labeled)

print("✅ Diff image saved.")

# ==========================================
# PRINT BRIGHTNESS STATS PER PHASE
# ==========================================

print("\n📈 Brightness stats per phase (mean pixel value, 0-255):")
print(f"   Phase 1 Original  : {np.mean(phase1):.1f}")
print(f"   Phase 2 Denoise   : {np.mean(phase2):.1f}")
print(f"   Phase 3 Gamma     : {np.mean(phase3):.1f}")
print(f"   Phase 4 CLAHE     : {np.mean(phase4):.1f}")
print(f"   Phase 5 Upscale   : {np.mean(phase5):.1f}  (same as phase 4, just larger)")

print("\n📁 Output folder contents:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) // 1024
    print(f"   {f}  ({size_kb} KB)")

print("\n✅ Done. Open the folder to inspect results:")
print(f"   {OUTPUT_DIR}")