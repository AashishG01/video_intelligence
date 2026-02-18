import cv2
import threading
import time

class CameraStream:
    def __init__(self, src, cam_id):
        self.cam_id = cam_id
        self.src = src
        self.stopped = False
        self.frame = None
        self.last_process_time = time.time()
        
        print(f"📡 Testing connection to {cam_id}...")
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        success, frame = self.stream.read()
        if not success:
            print(f"⚠️  WARNING: {cam_id} is unreachable.")
        else:
            self.frame = frame
            print(f"✅ {cam_id} is Streaming.")

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                time.sleep(1)
                self.stream = cv2.VideoCapture(self.src)
                continue
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()