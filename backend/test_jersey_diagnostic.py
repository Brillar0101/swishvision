"""
Diagnostic test for jersey number detection
Run this to see exactly where things are slow
"""
import time
import cv2
import numpy as np

print("="*60)
print("Jersey Detection Diagnostic")
print("="*60)

# Test 1: Load a frame
print("\n[1] Loading test frame...")
start = time.time()
cap = cv2.VideoCapture("../test_videos/test_game.mp4")
ret, frame = cap.read()
cap.release()
print(f"    Frame loaded: {frame.shape} in {time.time()-start:.2f}s")

# Test 2: Player detection model
print("\n[2] Loading player detection model...")
start = time.time()
from inference import get_model
player_model = get_model(model_id="basketball-players-fy4c2/3")
print(f"    Player model loaded in {time.time()-start:.2f}s")

print("\n[3] Running player detection...")
start = time.time()
result = player_model.infer(frame, confidence=0.3)[0]
print(f"    Detection done in {time.time()-start:.2f}s")
print(f"    Found {len(result.predictions)} detections")

# Test 3: Number box detection model
print("\n[4] Loading number box detection model...")
start = time.time()
number_model = get_model(model_id="basketball-jersey-numbers-ocr/3")
print(f"    Number model loaded in {time.time()-start:.2f}s")

print("\n[5] Running number box detection...")
start = time.time()
number_result = number_model.infer(frame, confidence=0.3)[0]
print(f"    Number detection done in {time.time()-start:.2f}s")
print(f"    Found {len(number_result.predictions)} number boxes")

# Test 4: OCR model - THIS IS LIKELY THE SLOW PART
print("\n[6] Loading OCR model (SmolVLM2 - this may be slow)...")
start = time.time()
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"    Using device: {device}")
    
    ocr_processor = AutoProcessor.from_pretrained(
        "mgivney/jersey-number-ocr",
        trust_remote_code=True
    )
    print(f"    Processor loaded in {time.time()-start:.2f}s")
    
    start = time.time()
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        "mgivney/jersey-number-ocr",
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(device)
    print(f"    OCR model loaded in {time.time()-start:.2f}s")
    
    # Test OCR on a small crop
    print("\n[7] Testing OCR inference on a crop...")
    if len(number_result.predictions) > 0:
        pred = number_result.predictions[0]
        x1 = int(pred.x - pred.width/2)
        y1 = int(pred.y - pred.height/2)
        x2 = int(pred.x + pred.width/2)
        y2 = int(pred.y + pred.height/2)
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        
        if crop.size > 0:
            from PIL import Image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            
            start = time.time()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "What is the jersey number in this image? Reply with only the number."}
            ]}]
            
            inputs = ocr_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)
            
            print(f"    Input prep done in {time.time()-start:.2f}s")
            
            start = time.time()
            with torch.no_grad():
                outputs = ocr_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            print(f"    OCR inference done in {time.time()-start:.2f}s")
            
            result_text = ocr_processor.decode(outputs[0], skip_special_tokens=True)
            print(f"    Result: {result_text}")
    else:
        print("    No number boxes found to test OCR")
        
except Exception as e:
    print(f"    ERROR: {e}")

print("\n" + "="*60)
print("Diagnostic complete")
print("="*60)
