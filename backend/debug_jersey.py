"""
Jersey Number Detection - Debug Version
"""
import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("Jersey Number Detection - DEBUG")
print("="*60)
print(f"Python: {sys.version}")

try:
    print("\n[1] Importing cv2...")
    import cv2
    print("    OK")
    
    print("\n[2] Importing numpy...")
    import numpy as np
    print("    OK")
    
    print("\n[3] Importing torch...")
    import torch
    print(f"    OK - torch {torch.__version__}")
    print(f"    MPS available: {torch.backends.mps.is_available()}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    
    print("\n[4] Loading video...")
    video_path = "../test_videos/test_game.mp4"
    if not os.path.exists(video_path):
        print(f"    ERROR: Video not found at {video_path}")
        print(f"    Current dir: {os.getcwd()}")
        print(f"    Looking for alternatives...")
        for root, dirs, files in os.walk(".."):
            for f in files:
                if f.endswith(".mp4"):
                    print(f"    Found: {os.path.join(root, f)}")
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("    ERROR: Could not read frame from video")
        sys.exit(1)
    print(f"    OK - Frame shape: {frame.shape}")
    
    print("\n[5] Importing inference...")
    from inference import get_model
    print("    OK")
    
    print("\n[6] Loading player model...")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("    ERROR: ROBOFLOW_API_KEY not set in .env")
        sys.exit(1)
    print(f"    API key found: {api_key[:5]}...{api_key[-3:]}")
    
    player_model = get_model(
        model_id="basketball-player-detection-3-ycjdo/4",
        api_key=api_key
    )
    print("    OK")
    
    print("\n[7] Running player detection...")
    result = player_model.infer(frame, confidence=0.3)[0]
    print(f"    OK - Found {len(result.predictions)} detections")
    
    # Get first player box
    player_box = None
    for pred in result.predictions:
        if pred.class_id == 3:
            x1 = int(pred.x - pred.width/2)
            y1 = int(pred.y - pred.height/2)
            x2 = int(pred.x + pred.width/2)
            y2 = int(pred.y + pred.height/2)
            player_box = [x1, y1, x2, y2]
            break
    
    if player_box is None:
        print("    WARNING: No players found, using center crop")
        h, w = frame.shape[:2]
        player_box = [w//4, h//4, w//2, h//2]
    
    print(f"    Player box: {player_box}")
    
    # Get jersey crop (upper body)
    x1, y1, x2, y2 = player_box
    crop_h = (y2 - y1) // 2
    crop = frame[y1:y1+crop_h, x1:x2]
    print(f"    Crop shape: {crop.shape}")
    
    cv2.imwrite("debug_crop.jpg", crop)
    print("    Saved debug_crop.jpg")
    
    print("\n[8] Importing transformers...")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print("    OK")
    
    print("\n[9] Loading SmolVLM2 processor...")
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        trust_remote_code=True
    )
    print("    OK")
    
    print("\n[10] Loading SmolVLM2 model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"     Using device: {device}")
    
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)
    print("    OK")
    
    print("\n[11] Preparing OCR input...")
    from PIL import Image
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "What jersey number is shown? Reply with only the number."}
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
    print("    OK")
    
    print("\n[12] Running OCR inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True
        )
    print("    OK")
    
    print("\n[13] Decoding result...")
    result_text = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"    Raw: {result_text}")
    
    import re
    numbers = re.findall(r'\d+', result_text)
    if numbers:
        print(f"    Detected number: {numbers[-1]}")
    else:
        print("    No number detected")
    
    print("\n" + "="*60)
    print("SUCCESS - All steps completed!")
    print("="*60)

except Exception as e:
    print(f"\n\nERROR at step above:")
    print(f"    {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
