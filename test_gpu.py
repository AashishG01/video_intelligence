import sys
import time
import numpy as np
import cv2
from insightface.app import FaceAnalysis

print("🔍 initializing InsightFace...")
print("------------------------------------------------")

# 1. Force CUDA Provider
# We explicitly ask for CUDA.
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Strict Verification
try:
    # InsightFace loads several models (detection, gender, age, recognition).
    # We check the 'detection' model's session to see what is actually running.
    detect_model = app.models.get('detection')
    if detect_model is None:
        print("❌ Error: Detection model failed to load completely.")
        sys.exit(1)

    active_providers = detect_model.session.get_providers()
    print(f"📢 Active Providers found: {active_providers}")

    if 'CUDAExecutionProvider' in active_providers:
        print("\n✅ SUCCESS: GPU is ACTIVE and READY!")
        print("   You are running on NVIDIA CUDA.")
    else:
        print("\n❌ FAILURE: System fell back to CPU.")
        print("   Reason: NVIDIA Drivers or CUDA Toolkit might be missing.")
        print("   Solution: Check 'nvidia-smi' in terminal.")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")

print("------------------------------------------------")