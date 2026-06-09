import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5
)


with HandLandmarker.create_from_options(options) as detector:

    input_dir = r"D:\Education\Computer Visign\some small projects\Sign langues version 4\handdata"

    output_dir = "finalannotated_data"
    no_detect_dir = os.path.join(output_dir, "no_hands")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(no_detect_dir, exist_ok=True)

    def process_image(img_path, save_path, no_detect_path):
        img = cv2.imread(img_path)
        if img is None:
            return False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)
        if len(detection_result.hand_landmarks) > 0:
            cv2.imwrite(save_path, img)
            return True
        else:
            cv2.imwrite(no_detect_path, img)
            return False
    total, detected = 0, 0

    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue
        save_folder = os.path.join(output_dir, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        for img_name in os.listdir(folder_path):
            total += 1
            img_path = os.path.join(folder_path, img_name)
            save_path = os.path.join(save_folder, img_name)
            no_detect_path = os.path.join(no_detect_dir, f"{folder_name}_{img_name}")

            if process_image(img_path, save_path, no_detect_path):
                detected += 1


    print(f"Done! \nDetected hands in {detected}/{total} images.")
    print(f"Images with no detection saved in: {no_detect_dir}")