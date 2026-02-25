import cv2
import numpy as np

# RTSP URLs for the four cameras
RTSP_URLS = [
    "rtsp://admin:admin@172.16.0.151:554/live.sdp",
    "rtsp://admin:admin@172.16.0.152:554/live.sdp",
    "rtsp://admin:123456@172.16.0.161:554/live.sdp",
    "rtsp://admin:Admin@123@172.16.0.162:554/live.sdp"
]

def create_placeholder_frame(width=640, height=480):
    """Create a black placeholder frame for failed streams."""
    return np.zeros((height, width, 3), dtype=np.uint8)

def main():
    # Initialize captures for all URLs
    caps = []
    for i, url in enumerate(RTSP_URLS):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Error: Could not open video stream at {url} (Camera {i+1})")
            caps.append(None)  # Placeholder for failed stream
        else:
            caps.append(cap)
            print(f"Stream {i+1} opened successfully.")

    # Check if at least one stream is open
    if all(cap is None for cap in caps):
        print("Error: No streams could be opened. Check network and credentials.")
        return

    print("2x2 Grid started. Press 'q' to quit.")

    # Define grid dimensions (adjust as needed for your display)
    frame_width = 640
    frame_height = 480
    grid_width = frame_width * 2
    grid_height = frame_height * 2

    while True:
        frames = []
        for i, cap in enumerate(caps):
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    # Resize frame to fit quadrant
                    frame = cv2.resize(frame, (frame_width, frame_height))
                else:
                    print(f"Warning: Failed to read frame from Camera {i+1}. Using placeholder.")
                    frame = create_placeholder_frame(frame_width, frame_height)
            else:
                frame = create_placeholder_frame(frame_width, frame_height)
            frames.append(frame)

        # Arrange into 2x2 grid: Top row (frames 0-1), Bottom row (frames 2-3)
        top_row = np.hstack([frames[0], frames[1]])
        bottom_row = np.hstack([frames[2], frames[3]])
        grid = np.vstack([top_row, bottom_row])

        # Display the grid
        cv2.imshow('2x2 RTSP Grid', grid)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    for cap in caps:
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()