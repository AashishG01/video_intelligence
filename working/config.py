# config.py
import os

RTSP_URL_1 = "rtsp://admin:admin@172.16.0.151:554/live.sdp" 
RTSP_URL_2 = "rtsp://admin:admin@172.16.0.152:554/live.sdp" 

# FIX 1: Lower interval to catch the head turning (creates the "breadcrumb" chain)
PROCESS_INTERVAL = 0.2 

# FIX 2: Relax the threshold slightly to allow for angle differences
FACE_MATCH_THRESHOLD = 0.50 

os.environ["ORT_LOGGING_LEVEL"] = "3" 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "live_video_db")
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "captured_faces")