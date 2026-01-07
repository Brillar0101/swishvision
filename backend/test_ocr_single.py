"""Test OCR on just one number box"""
import cv2
import numpy as np
import supervision as sv
from inference import get_model

# Load frame
cap = cv2.VideoCapture("../test_videos/test_game.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

# Detect number boxes
print("Detecting number boxes...")
model = get_model(model_id="basketball-player-detection-3-ycjdo/4")
result = model.infer(frame, confidence=0.4)[0]
detections = sv.Detections.from_inference(result)
numbers = detections[detections.class_id == 2]

print(f"Found {len(numbers)} number boxes")

if len(numbers) > 0:
    # Crop first number box
    x1, y1, x2, y2 = numbers.xyxy[0].astype(int)
    crop = frame[max(0,y1-10):y2+10, max(0,x1-10):x2+10]
    crop_resized = cv2.resize(crop, (224, 224))
    
    cv2.imwrite("../outputs/number_crop.jpg", crop_resized)
    print("Saved crop: ../outputs/number_crop.jpg")
    
    # Try OCR
    print("Loading OCR model (may take a moment)...")
    ocr_model = get_model(model_id="basketball-jersey-numbers-ocr/3")
    
    print("Running OCR...")
    result = ocr_model.predict(crop_resized, "Read the number.")
    print(f"OCR Result: {result}")
