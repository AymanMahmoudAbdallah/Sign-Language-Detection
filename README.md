# Sign Language Detection

A real-time Sign Language Recognition system developed using TensorFlow, OpenCV, and MediaPipe. The project uses a custom-built dataset collected manually, extracts hand landmark features, trains a deep learning model, and performs live gesture recognition through a webcam.

## Overview

This project aims to recognize sign language gestures in real time using computer vision and machine learning techniques.

The complete pipeline includes:

1. Custom dataset collection.
2. Hand landmark extraction using MediaPipe.
3. Feature preprocessing and storage.
4. TensorFlow model training.
5. Real-time gesture classification using OpenCV.
6. Multi-threaded inference for improved responsiveness.

---

## Features

- Real-time sign language recognition
- Custom dataset collected and labeled manually
- Hand landmark feature extraction using MediaPipe
- TensorFlow-based classification model
- Webcam integration using OpenCV
- Multi-threaded architecture for smoother performance
- Low-latency predictions

---

## Technologies Used

- Python
- TensorFlow
- OpenCV
- MediaPipe
- NumPy
- Pandas

---

## Project Pipeline

### 1. Dataset Collection

A custom dataset was created by capturing hand gesture images manually for each sign language class.

### 2. Feature Extraction

Instead of training directly on raw images, MediaPipe was used to detect hand landmarks and extract meaningful hand geometry features.

These extracted features were stored in structured files and used as the training dataset.

### 3. Model Training

The extracted landmark features were used to train a TensorFlow classification model capable of recognizing different sign language gestures.

### 4. Real-Time Inference

OpenCV captures webcam frames while MediaPipe extracts hand landmarks in real time.

The trained TensorFlow model then predicts the gesture and displays the result instantly.

### 5. Multi-Threading Optimization

To improve responsiveness and reduce latency:

- A dedicated thread handles TensorFlow inference.
- A separate thread manages video capture and frame processing.

This architecture allows smoother real-time predictions without blocking the main application flow.

---

## System Architecture

```text
Webcam
   │
   ▼
OpenCV Frame Capture
   │
   ▼
MediaPipe Hand Detection
   │
   ▼
Feature Extraction
   │
   ▼
TensorFlow Model
   │
   ▼
Gesture Prediction
   │
   ▼
Display Result
```

---

## Challenges Solved

- Building a custom dataset from scratch.
- Converting hand landmarks into trainable numerical features.
- Achieving real-time inference performance.
- Synchronizing TensorFlow predictions with live video processing using multi-threading.

---

## Future Improvements

- Support full words and sentences.
- Text-to-speech integration.
- Arabic Sign Language support.
- Larger dataset for higher accuracy.
- Web and mobile deployment.

---

## Author

**Ayman Mahmoud Abdallah**

Computer Science Student | AI Enthusiast

GitHub:
https://github.com/AymanMahmoudAbdallah

---

## License

This project is intended for educational and research purposes.
