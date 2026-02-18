import cv2

# RTSP URL provided
RTSP_URL = "rtsp://admin:admin@172.16.0.151:554/live.sdp"

def main():
    # Initialize capture
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"Error: Could not open video stream at {RTSP_URL}")
        print("Ensure your computer is on the same network (172.16.x.x) and credentials are correct.")
        return

    print("Stream started. Press 'q' to quit.")

    while True:
        # Read frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to retrieve frame (stream ended or connection lost).")
            break

        # Display the resulting frame
        cv2.imshow('RTSP Stream', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()