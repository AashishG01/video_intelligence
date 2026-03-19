Testing files 

extract_image.py
3fps 
CONFIDENCE_THRESHOLD = 0.60
MIN_FACE_SIZE = 50
det_size = (1024, 1024)

/home/user/Desktop/video surveillence/Sample_videos/Export__Rly Station-Towards Amisha Hotel Right_Friday February 20 2026114208  608d3f6.avi
this video -> 958 faces detected

/home/user/Desktop/video surveillence/Sample_videos/Export__Rly Station Parking Area-Entry_Wednesday February 18 202694311  aa4cf13.avi
this video -> 814 faces detected 

/home/user/Desktop/video surveillence/Sample_videos/Export__MajuraGate-Towards sagrampura_Wednesday February 18 202695109  1a65bd9.avi
this video -> 536 faces detected

All this faces are stored in testing image folder 

/home/user/Desktop/video surveillence/Sample_videos/Export__Mahidharpura Nr Temple Thoba Sheri_Friday February 20 2026114459  a430615.avi
this video -> 3246 faces detected
all this faces are stored in extreme image 

for this video we made some changes in configuration 
That is 
# 1. Lower the size limit drastically (catch faces as small as 15x15 pixels)
MIN_FACE_SIZE = 15

# 2. Lower the confidence threshold (Warning: This will cause some false positives)
CONFIDENCE_THRESHOLD = 0.35 

# 3. OVERCLOCK THE RESOLUTION
# Change the det_size in the app.prepare() function to 1920x1920 or higher.
# This forces the GPU to blow the image up massively before scanning it.
app.prepare(ctx_id=0, det_size=(1920, 1920))

This might be the limitation of our system as wehave reduced the pixel for faces to 15x15 -> confidence threshold to 0.35 for to create embeddings and matching it will be diffcult as it will give very much miss match 



accuracy testing code 
test_accuracy.py 

Algo  --> 

Mugshot Override (det_thresh = 0.10)
Lowers the face detection confidence threshold from 60% to 10%.
Prevents tightly cropped face images (eyes, nose, mouth only) from being rejected.
Forces the AI to accept the image as a face and generate a 512-D embedding vector.

Automated Test Loop
Loads each cropped face image from a test folder.
Convrts the face into a 512-dimensional vector using the same logic as production.
Searches ChromaDB for the closest matching identity.

Compares the distance score:
✅ Match if distance < 0.50
❌ Mismatch if distance ≥ 0.50
Repeats this process for every image in the folder.

Final Scorecard
Tallies correct vs incorrect matches across all test images.
Calculates overall system accuracy (e.g., 85 correct out of 100 = 85.00%).
Prints a clear accuracy report in the terminal for fast, UI-free evaluation.

using testing data from test image 
and search through the database of live_video_db

Match_threshold = 50
system accuracy = 79.25

Match_threshold = 40 -> strict checking 
system accuracy = 59.71

Match_threshold = 60 -> loose checking 
system accuracy = 90.73

Match_threshold = 55
system accuracy = 86.48

Match_threshold = 45
system accuracy = 69.93


I have created a file for a visual audit (test_visual_audit.py) -- for Match_thershold = 55 based on visual audit it is recommended to go threshold = 57 or 58
folder = visual_audit_results



code for embedding generation and storage in db is final_code_4cam.py
backend -> backend/api.py  -> uvicorn api:app --host 0.0.0.0 --port 8000 --reload
frontend -> npm run dev 
venv activation -> source venv/bin/activate


extreme image testing --> pixels are very small 
Match_threshold = 45
system accuracy -> 61.03

Match_threshold = 55
system accuracy = 83.98




What we can do next 

1. Person tracking 
2. multi frame averaging for better embeddings 
3. super image resolution 




1. night video testing 
2. sorting 
3. new user interface 
4. memory usage and cpu usage parameters 



night_facecollector.py
It reads a night video, enhances each frame to improve visibility, detects faces using AI, and saves each unique person into their own folder in a database. Video frame → enhance → detect faces → filter weak ones → check DB → save to person's folder


Enhancements 
Denoise 
Gamma
Clahe
upscale 


night enhancement results 

test images were also enhanced 
accuracy -> 34% 
threshold -> 0.55

==============================
📊 BENCHMARK REPORT
==============================
Runtime              : 40.82 sec
Total Frames Read    : 1510
Frames Processed     : 50
Processing FPS       : 1.23
Frame Latency        : 0.8163 sec
Faces Detected       : 2876
Faces Saved          : 217

🌙 NIGHT ENHANCEMENT  (avg ms per frame)
  Denoise            : 10.54 ms
  Gamma correction   : 1.62 ms
  CLAHE              : 13.67 ms
  Upscale (1.5x)      : 8.22 ms
  ─────────────────────────────
  Total enhancement  : 34.06 ms / frame

🖥 CPU / RAM
Average CPU Usage    : 3.78%
Average RAM Usage    : 11.09%

🎮 GPU Metrics
Average GPU Usage    : 33.76%
Average GPU Memory   : 3.01 GB
Average GPU Power    : 171.34 W

📹 ESTIMATED CAPACITY
Camera FPS                    : 20.0
Frames processed per camera   : 0.67
Estimated cameras per GPU     : 1
Camera mode                   : High-angle

✅ System Closed

============================================================
📊 FINAL NIGHT PIPELINE ACCURACY REPORT
============================================================
Total Test Images Scanned  : 4658
------------------------------------------------------------
✅ Successful Matches       : 4313
❌ Threshold Mismatches     : 148
❌ No Face Detected         : 197
------------------------------------------------------------
🎯 Accuracy (all images)    : 92.59%
🎯 Accuracy (reached DB)    : 96.68%
============================================================




Notes day -> 19 march 2026

Final checking for streams -> stream_viewer.py 

Final 4 cam working and saving data in Database folder -> final_code_4cam.py 

Night_vision surveillience_with_enhancement -> test_night_single.py

benchmark code for 4 live camera -> test_4camlive_benchmark.py




night enhancement score/accuracy 

============================================================
📊 FINAL PIPELINE ACCURACY REPORT
============================================================
Total Test Images Scanned: 3368
------------------------------------------------------------
✅ Successful Matches:      3140
❌ Failed / Mismatches:     228
------------------------------------------------------------
🎯 System Accuracy Rate:    93.23%
============================================================
