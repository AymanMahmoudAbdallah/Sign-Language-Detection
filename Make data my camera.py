import cv2  
import os
import time
from datetime import datetime
    
# Settings
SAVE_DIR = "photos"
MAX_PHOTOS = 120
INTERVAL_SECONDS = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot access the camera.")

print("Press 'q' to start")

capturing = False
photo_count = 0
last_saved = 0.0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame , 1)
        if not ret:
            print("Failed to read frame from camera.")
            break

        cv2.imshow("Camera (press q to start/stop)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') and not capturing:
            print("Started capturing...")
            capturing = True
            photo_count = 0
            last_saved = time.time()

        if capturing:
            now = time.time()
            if now - last_saved >= INTERVAL_SECONDS:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = os.path.join(SAVE_DIR, f"photo_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                photo_count += 1
                print(f"[{photo_count}] Saved: {filename}")
                last_saved = now

                if photo_count >= MAX_PHOTOS:
                    print("Captured 120 photos. Stopping...")
                    capturing = False

        if key == 27:  # ESC
            print("Exited by user.")
            break

except KeyboardInterrupt:
    print("\nStopped manually (KeyboardInterrupt).")

finally:
    cap.release()
    cv2.destroyAllWindows()
