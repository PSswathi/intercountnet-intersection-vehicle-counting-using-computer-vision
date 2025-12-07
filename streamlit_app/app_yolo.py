import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

MODEL_PATH = "best.pt"
CLASS_NAMES = ["bicycle", "bus", "car", "license-plate", "motorcycle"]
CONF_THRESHOLD = 0.5
FRAME_SKIP = 5
RESIZE = (640, 480)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("🚦 YOLOv8 Vehicle Detection & Counting")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp.mp4")
    stframe = st.empty()
    counts = defaultdict(int)

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, RESIZE)

        # YOLO inference
        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = CLASS_NAMES[cls]
            counts[class_name] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

        stframe.image(frame, channels="BGR")

    st.success("Completed video processing!")
    st.write("### Final Counts")
    st.json(counts)

    cap.release()