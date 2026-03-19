Running 4cam live for 2 mins at night only detecting around 2 people 

============================================================
📊 BENCHMARK REPORT — LIVE RTSP (4 Cameras)
============================================================
Runtime                  : 120.23 sec
Cameras Online           : 4 / 4
Cameras Offline          : 0 / 4
Frames Processed (total) : 445
Avg Processing Rate      : 3.70 frames/sec
------------------------------------------------------------
Faces Detected           : 447
Faces Matched (existing) : 434
New Persons Created      : 0
Faces Saved              : 434
------------------------------------------------------------
Avg AI Inference Time    : 34.85 ms / frame
------------------------------------------------------------
Avg CPU Usage            : 2.93%
Avg RAM Usage            : 12.10%
Avg GPU Usage            : 9.41%
Avg GPU Memory           : 3.20 GB
Avg GPU Power            : 108.25 W
------------------------------------------------------------
Processing Interval      : 1.0 sec
Estimated Max Cameras    : 28
============================================================


benchmark for the one video of 5mins

==============================
📊 BENCHMARK REPORT
==============================
Runtime: 34.77 sec
Total Frames Read: 6020
Frames Processed: 200
Processing FPS: 5.75
Frame Latency: 0.1739 sec
Faces Detected: 485
Faces Saved: 427

🖥 CPU / RAM
Average CPU Usage: 6.28%
Average RAM Usage: 8.33%

🎮 GPU Metrics
Average GPU Usage: 17.91%
Average GPU Memory: 2.88 GB
Average GPU Power: 108.65 W

📹 ESTIMATED CAPACITY
Camera FPS: 20.0
Frames processed per camera: 0.67
Estimated cameras per GPU: 8


benchmark for night data night enhancement results 

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


