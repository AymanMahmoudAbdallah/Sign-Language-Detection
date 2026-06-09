import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

def get_processed_landmarks(img, detector):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_lms = detection_result.hand_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])

        points = points - points[0]

        max_value = np.abs(points).max()
        if max_value > 0:
            points = points / max_value

        return points.flatten().tolist()
    return None

# ─── المسارات ─────────────────────────────────────────────────
tdata_path = r"D:\Education\Computer Visign\some small projects\Sign langues version 4\finalannotated_data"
final_outputdata = []
final_outputlabel = []

print("Starting Data Extraction...")

with vision.HandLandmarker.create_from_options(options) as detector:
    classes = sorted(
        [d for d in os.listdir(tdata_path)
         if os.path.isdir(os.path.join(tdata_path, d))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    print("Class mapping:")
    for idx, name in enumerate(classes):
        print(f"  Label {idx} → Class '{name}'")

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(tdata_path, class_name)

        print(f"Processing Class: {class_name} (label={class_index})")

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            land = get_processed_landmarks(img, detector)
            if land:
                final_outputdata.append(land)
                final_outputlabel.append(class_index)

np.savetxt('landmarks_data.txt', np.array(final_outputdata))
np.savetxt('landmarks_labels.txt', np.array(final_outputlabel))

class_mapping = {i: name for i, name in enumerate(classes)}
np.save('class_mapping.npy', class_mapping)

print("-" * 40)
print(f"Done! Saved {len(final_outputdata)} samples.")
print(f"Classes: {class_mapping}")