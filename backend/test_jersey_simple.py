"""Test just number box detection - no OCR"""
import cv2
import numpy as np
import supervision as sv
from inference import get_model

# Load one frame
cap = cv2.VideoCapture("../test_videos/test_game.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()
print(f"Frame loaded: {frame.shape}")

# Load player detection model
print("Loading model...")
model = get_model(model_id="basketball-player-detection-3-ycjdo/4")

# Detect
print("Running detection...")
result = model.infer(frame, confidence=0.4)[0]
detections = sv.Detections.from_inference(result)

# Filter by class
NUMBER_CLASS_ID = 2
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]

numbers = detections[detections.class_id == NUMBER_CLASS_ID]
players = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]

print(f"Players: {len(players)}, Number boxes: {len(numbers)}")

# Draw
annotated = frame.copy()
for i in range(len(players)):
    x1, y1, x2, y2 = players.xyxy[i].astype(int)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

for i in range(len(numbers)):
    x1, y1, x2, y2 = numbers.xyxy[i].astype(int)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(annotated, "NUM", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imwrite("../outputs/test_number_boxes.jpg", annotated)
print("Saved: ../outputs/test_number_boxes.jpg")
