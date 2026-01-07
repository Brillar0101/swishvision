"""
SmolVLM2 Jersey OCR Diagnostic
This will show exactly where the slowdown is
"""
import time
import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("SmolVLM2 Jersey OCR Diagnostic")
print("="*60)

# Step 1: Load frame and get a player crop
print("\n[1] Loading test frame...")
cap = cv2.VideoCapture("../test_videos/test_game.mp4")
ret, frame = cap.read()
cap.release()
print(f"    Frame: {frame.shape}")

# Step 2: Get a player detection
print("\n[2] Getting player detection...")
from inference import get_model
player_model = get_model(
    model_id="basketball-player-detection-3-ycjdo/4",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)
result = player_model.infer(frame, confidence=0.3)[0]

# Find first player
player_crop = None
for pred in result.predictions:
    if pred.class_id == 3:
        x1 = int(pred.x - pred.width/2)
        y1 = int(pred.y - pred.height/2)
        x2 = int(pred.x + pred.width/2)
        y2 = int(pred.y + pred.height/2)
        
        # Get upper body (jersey area)
        h = y2 - y1
        crop_y2 = y1 + int(h * 0.5)
        player_crop = frame[max(0,y1):crop_y2, max(0,x1):x2]
        
        if player_crop.size > 0:
            cv2.imwrite("debug_jersey_crop.jpg", player_crop)
            print(f"    Got player crop: {player_crop.shape}")
            print("    Saved to debug_jersey_crop.jpg")
            break

if player_crop is None:
    print("    ERROR: No player found")
    exit()

# Step 3: Load SmolVLM2
print("\n[3] Importing transformers...")
start = time.time()
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
print(f"    Imports done in {time.time()-start:.2f}s")

print("\n[4] Checking device...")
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"    Using device: {device}")

print("\n[5] Loading SmolVLM2 processor...")
print("    (This downloads ~500MB on first run)")
start = time.time()
processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    trust_remote_code=True
)
print(f"    Processor loaded in {time.time()-start:.2f}s")

print("\n[6] Loading SmolVLM2 model...")
print("    (This downloads ~500MB on first run)")
start = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
).to(device)
print(f"    Model loaded in {time.time()-start:.2f}s")

# Step 4: Prepare input
print("\n[7] Preparing input...")
start = time.time()
crop_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(crop_rgb)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "What jersey number is shown? Reply with only the number, nothing else."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)
print(f"    Input prepared in {time.time()-start:.2f}s")

# Step 5: Run inference
print("\n[8] Running SmolVLM2 inference...")
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        use_cache=True
    )
inference_time = time.time() - start
print(f"    Inference done in {inference_time:.2f}s")

# Step 6: Decode output
print("\n[9] Decoding output...")
result_text = processor.decode(outputs[0], skip_special_tokens=True)
print(f"    Raw output: {result_text}")

# Extract just the number
import re
numbers = re.findall(r'\d+', result_text)
if numbers:
    jersey_number = numbers[-1]  # Take last number (usually the answer)
    print(f"    Detected jersey number: {jersey_number}")
else:
    print("    No number detected")

print("\n" + "="*60)
print(f"SUMMARY: SmolVLM2 inference takes {inference_time:.2f}s per crop")
print(f"For 10 players = ~{inference_time * 10:.0f}s per frame")
print("="*60)

if inference_time > 5:
    print("\n⚠️  WARNING: Inference is slow!")
    print("   Options:")
    print("   1. Use GPU (RunPod) instead of Mac")
    print("   2. Only run OCR every N frames")
    print("   3. Use smaller model or EasyOCR")
