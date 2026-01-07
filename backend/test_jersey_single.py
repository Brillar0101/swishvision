"""Test jersey number detection on a single frame"""
import cv2
import numpy as np
import supervision as sv
from app.ml.player_detector import PlayerDetector
from app.ml.jersey_number_detector import JerseyNumberDetector

# Load one frame
cap = cv2.VideoCapture("../test_videos/test_game.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)  # Frame 50
ret, frame = cap.read()
cap.release()

print("Testing jersey number detection on single frame...")

# Detect players
print("1. Detecting players...")
player_detector = PlayerDetector()
detections = player_detector.detect(frame)
print(f"   Found {len(detections)} detections")

# Detect jersey numbers
print("2. Detecting jersey numbers...")
jersey_detector = JerseyNumberDetector()
numbers = jersey_detector.process_frame(frame, detections)
print(f"   Found numbers: {numbers}")

# Draw results
annotated = frame.copy()
for i in range(len(detections)):
    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add jersey number if found
    num = numbers.get(i, "?")
    label = f"#{num}"
    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite("../outputs/test_jersey_single.jpg", annotated)
print(f"\nSaved: ../outputs/test_jersey_single.jpg")
