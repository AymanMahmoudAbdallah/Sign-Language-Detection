import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
import time
import threading
from collections import Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

latest_resulthand = None
result_lock = threading.Lock()

outputs_keys = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: "G",
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 38: 'Delete', 39: 'Space',
    None: 'Unknown'
}

prediction_buffer = []
buffer_size = 12
stable_letter = ""
final_word = ""
full_sentence = ""
last_addition_time = time.time()
hand_last_seen_time = time.time()
cooldown_duration = 0.8

def result_callbackhand(result, output_image, timestamp_ms):
    global latest_resulthand
    with result_lock:
        latest_resulthand = result

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callbackhand,
    num_hands=1,
)
detectorhand = vision.HandLandmarker.create_from_options(options)
model = tf.keras.models.load_model("best_sign_language_model.keras")

@tf.function(input_signature=[tf.TensorSpec(shape=(1, 63), dtype=tf.float32)])
def fast_predict(x):
    return model(x, training=False)

_ = fast_predict(tf.zeros((1, 63), dtype=tf.float32))
print("Model warmed up!")


cam = CameraStream(0)
#cam = cv2.VideoCapture('http://10.77.14.202:8080/video')
print("--- Started (Press 'x' to exit) ---")

fps_counter = 0
fps_start = time.time()
display_fps = 0

try:
    while True:
        ret, frame = cam.read()

        #frame = cv2.resize(frame , (800,600))
        #   frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE )


        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detectorhand.detect_async(mp_image, int(time.perf_counter() * 1000))

        with result_lock:
            current_result = latest_resulthand

        if current_result and current_result.hand_landmarks:
            hand_last_seen_time = time.time()
            hand_landmarks = current_result.hand_landmarks[0]

            raw_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
            processed_points = raw_points - raw_points[0]
            max_val = np.abs(processed_points).max()
            if max_val > 0:
                processed_points /= max_val

            input_data = tf.constant(processed_points.flatten().reshape(1, -1), dtype=tf.float32)
            preds = fast_predict(input_data)
            pred_class = int(np.argmax(preds.numpy()))
            current_pred = outputs_keys.get(pred_class, 'Unknown')

            prediction_buffer.append(current_pred)
            if len(prediction_buffer) > buffer_size:
                prediction_buffer.pop(0)

            most_common_letter, count = Counter(prediction_buffer).most_common(1)[0]
            if count > (buffer_size * 0.7):
                if most_common_letter != stable_letter:
                    stable_letter = most_common_letter
                    last_addition_time = time.time()

                if time.time() - last_addition_time > cooldown_duration:
                    if stable_letter == 'Space':
                        if final_word.strip():
                            full_sentence += final_word.strip() + " "
                            final_word = ""
                    elif stable_letter not in ['Nonex', 'Unknown', 'Delete']:
                        final_word += stable_letter

                    prediction_buffer.clear()
                    stable_letter = ""
                    last_addition_time = time.time()

            list_X = [int(lm.x * W) for lm in hand_landmarks]
            list_Y = [int(lm.y * H) for lm in hand_landmarks]
            cv2.rectangle(frame,
                (min(list_X)-20, min(list_Y)-20),
                (max(list_X)+20, max(list_Y)+20),
                (0, 255, 0), 2)
            cv2.putText(frame, f"Sign: {current_pred}",
                (min(list_X), min(list_Y)-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        else:
            if final_word.strip() and (time.time() - hand_last_seen_time > 1.0):
                full_sentence += final_word.strip() + " "
                final_word = ""
                prediction_buffer.clear()
                stable_letter = ""

        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            display_fps = fps_counter
            fps_counter = 0
            fps_start = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'): break
        elif key == ord('c'): final_word = ""; full_sentence = ""
        elif key == ord('b'): final_word = final_word[:-1]
        elif key == ord('d'):
            if full_sentence.strip():
                words = full_sentence.strip().split()
                full_sentence = " ".join(words[:-1])
                if full_sentence: full_sentence += " "

        # --- UI ---
        cv2.rectangle(frame, (0, 0), (W, 50), (20, 20, 20), -1)
        cv2.putText(frame, f"SENTENCE: {full_sentence}", (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (0, H-65), (W, H), (45, 45, 45), -1)
        cv2.putText(frame, f"WORD: {final_word}", (20, H-20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 250, 0), 2)
        cv2.putText(frame, f"FPS: {display_fps}", (W-120, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

        cv2.imshow('AI Sign Language', frame)

finally:
    cam.release()
    detectorhand.close()
    cv2.destroyAllWindows()