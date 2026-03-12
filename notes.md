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

