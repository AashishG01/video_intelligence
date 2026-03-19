import cv2
import numpy as np
import os
import threading

# Force TCP transport
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

RTSP_URLS = [
    "rtsp://admin:admin@172.16.0.151:554/live.sdp",
    "rtsp://admin:admin@172.16.0.152:554/live.sdp",
    "rtsp://admin:123456@172.16.0.161:554/live.sdp",
    "rtsp://admin:Admin@123@172.16.0.162:554/live.sdp"
]

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ==========================================
# HELPER — pehle define karo
# ==========================================
def create_placeholder_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    return np.zeros((height, width, 3), dtype=np.uint8)

# Ab buffer banao — function pehle define ho chuka hai
frame_buffer = [create_placeholder_frame() for _ in range(4)]
buffer_lock  = threading.Lock()

# ==========================================
# Per-camera capture thread
# ==========================================
def capture_thread(cam_idx, url):
    print(f"⏳ Connecting Camera {cam_idx + 1}...")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print(f"❌ Camera {cam_idx + 1} could not open: {url}")
        return

    print(f"✅ Camera {cam_idx + 1} connected.")

    # Skip until first valid keyframe
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            break

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"⚠️  Camera {cam_idx + 1}: Frame read failed, reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            continue

        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        with buffer_lock:
            frame_buffer[cam_idx] = resized

# ==========================================
# MAIN
# ==========================================
def main():
    threads = []
    for i, url in enumerate(RTSP_URLS):
        t = threading.Thread(target=capture_thread, args=(i, url), daemon=True)
        t.start()
        threads.append(t)

    print("🟢 2x2 Grid started. Press 'q' to quit.")

    while True:
        with buffer_lock:
            frames = [f.copy() for f in frame_buffer]

        top_row    = np.hstack([frames[0], frames[1]])
        bottom_row = np.hstack([frames[2], frames[3]])
        grid       = np.vstack([top_row, bottom_row])

        cv2.imshow('2x2 RTSP Grid', grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()