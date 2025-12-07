import streamlit as st
import torch
import cv2
import numpy as np
from collections import defaultdict, OrderedDict
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from scipy.spatial import distance as dist

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_faster_rcnn_vehicle_detector.pt")

CLASS_NAMES = ["bicycle", "bus", "car", "license-plate", "motorcycle"]
CONF_THRESHOLD = 0.5
FRAME_SKIP = 5                 # for speed
RESIZE = (640, 480)            # for speed

# ---------------- CENTROID TRACKER ----------------
class CentroidTracker:
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):

        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    num_classes = 6  # background + 5 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()
tracker = CentroidTracker(maxDisappeared=40)

# ---------------- UI ----------------
st.title("🚦 Unique Vehicle Detection & Counting (Faster R-CNN + Tracking)")
st.write("Upload a video to detect, track, and count each vehicle only ONCE")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:

    # Save video temporarily
    video_path = os.path.join(BASE_DIR, "temp_video.mp4")

    with open(video_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()
    counter_box = st.empty()

    # GLOBAL unique counts
    unique_ids_seen = set()
    unique_class_count = defaultdict(int)

    frame_id = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Skip frames for speed
        if frame_id % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, RESIZE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_tensor = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)

        with torch.no_grad():
            output = model([img_tensor])[0]

        boxes = output["boxes"].numpy()
        labels = output["labels"].numpy()
        scores = output["scores"].numpy()

        rects = []
        class_list = []

        for box, label, score in zip(boxes, labels, scores):

            if score < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.astype(int)
            rects.append((x1, y1, x2, y2))
            class_list.append(CLASS_NAMES[label - 1])

        objects = tracker.update(rects)

        for ((objectID, centroid), box, cls) in zip(objects.items(), rects, class_list):

            if objectID not in unique_ids_seen:
                unique_ids_seen.add(objectID)
                unique_class_count[cls] += 1

            x1, y1, x2, y2 = box

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"ID {objectID}: {cls}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

        # Show video
        stframe.image(frame, channels="BGR")

        # Show live total
        txt = "## ✅ Unique Vehicles So Far\n\n"
        txt += f"**Total: {len(unique_ids_seen)}**\n\n"
        for c in CLASS_NAMES:
            txt += f"- {c}: {unique_class_count[c]}\n"

        counter_box.markdown(txt)

    cap.release()

    # -------- FINAL SUMMARY --------
    st.success("✅ Video processing complete!")

    st.markdown("### 📊 FINAL UNIQUE VEHICLE COUNT (ENTIRE VIDEO)")

    final = f"**Total unique vehicles : {len(unique_ids_seen)}**\n\n"
    for c in CLASS_NAMES:
        final += f"- {c}: {unique_class_count[c]}\n"

    st.markdown(final)