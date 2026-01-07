"""
SwishVision - Simplified Tactical View Test
Based on Roboflow's basketball-ai notebook approach
Outputs: 3 sample frames with team labels and tactical court view
"""
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from tqdm import tqdm
import os

from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court
from sports import ViewTransformer, MeasurementUnit, TeamClassifier

# ============== CONFIGURATION ==============
VIDEO_PATH = '../test_videos/test_game.mp4'
OUTPUT_DIR = '../outputs/tactical_view'
MAX_SECONDS = 8
SAMPLE_FRAMES = 3

# Model IDs
PLAYER_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
KEYPOINT_MODEL_ID = "basketball-court-detection-2/14"

# Detection settings
PLAYER_CONFIDENCE = 0.4
PLAYER_IOU = 0.9
KEYPOINT_CONFIDENCE = 0.3
KEYPOINT_ANCHOR_CONFIDENCE = 0.5

# Class IDs
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]  # player variants
REFEREE_CLASS_IDS = [0, 1, 2]  # referee variants

# Team names
TEAM_NAMES = {
    0: "Team A",
    1: "Team B"
}

TEAM_COLORS = {
    "Team A": "#00FF00",  # Green
    "Team B": "#FF0000",  # Red
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("SwishVision - Tactical View Test")
    print("=" * 60)
    
    # Load models
    print("\n[1/7] Loading models...")
    player_model = get_model(model_id=PLAYER_MODEL_ID)
    keypoint_model = get_model(model_id=KEYPOINT_MODEL_ID)
    print("   Models loaded")
    
    # Load video frames
    print("\n[2/7] Loading video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_frames = int(fps * MAX_SECONDS)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"   Loaded {len(frames)} frames ({len(frames)/fps:.1f}s)")
    
    # Collect crops for team classification
    print("\n[3/7] Collecting player crops for team classification...")
    crops = []
    stride = 30  # Sample every 30 frames
    
    for i in range(0, len(frames), stride):
        frame = frames[i]
        result = player_model.infer(frame, confidence=PLAYER_CONFIDENCE, iou_threshold=PLAYER_IOU)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
        
        # Scale boxes to get center crops (jersey area)
        boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
        for box in boxes:
            crop = sv.crop_image(frame, box)
            if crop.size > 0:
                crops.append(crop)
    
    print(f"   Collected {len(crops)} player crops")
    
    # Train team classifier
    print("\n[4/7] Training team classifier...")
    team_classifier = TeamClassifier(device="cpu")  # Use CPU on Mac
    team_classifier.fit(crops)
    print("   Team classifier trained")
    
    # Court configuration
    config = CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)
    
    # Define annotators
    team_colors_palette = sv.ColorPalette.from_hex([
        TEAM_COLORS[TEAM_NAMES[0]],
        TEAM_COLORS[TEAM_NAMES[1]]
    ])
    
    team_box_annotator = sv.BoxAnnotator(
        color=team_colors_palette,
        thickness=2,
        color_lookup=sv.ColorLookup.INDEX
    )
    
    team_label_annotator = sv.LabelAnnotator(
        color=team_colors_palette,
        text_color=sv.Color.WHITE,
        color_lookup=sv.ColorLookup.INDEX
    )
    
    ref_box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFF00"]),
        thickness=2
    )
    
    # Process sample frames
    print("\n[5/7] Processing sample frames...")
    sample_indices = [int(i * (len(frames) - 1) / (SAMPLE_FRAMES - 1)) for i in range(SAMPLE_FRAMES)]
    
    for idx, frame_idx in enumerate(sample_indices):
        frame = frames[frame_idx]
        annotated = frame.copy()
        
        # Detect players
        result = player_model.infer(frame, confidence=PLAYER_CONFIDENCE, iou_threshold=PLAYER_IOU)[0]
        detections = sv.Detections.from_inference(result)
        
        # Split players and referees
        players = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
        refs = detections[np.isin(detections.class_id, REFEREE_CLASS_IDS)]
        
        # Classify teams for players
        if len(players) > 0:
            boxes = sv.scale_boxes(xyxy=players.xyxy, factor=0.4)
            player_crops = [sv.crop_image(frame, box) for box in boxes]
            teams = np.array(team_classifier.predict(player_crops))
            
            # Create labels
            labels = [TEAM_NAMES[t] for t in teams]
            
            # Annotate players
            annotated = team_box_annotator.annotate(
                scene=annotated,
                detections=players,
                custom_color_lookup=teams
            )
            annotated = team_label_annotator.annotate(
                scene=annotated,
                detections=players,
                labels=labels,
                custom_color_lookup=teams
            )
        else:
            teams = np.array([])
        
        # Annotate referees
        if len(refs) > 0:
            ref_labels = ["Referee"] * len(refs)
            annotated = ref_box_annotator.annotate(scene=annotated, detections=refs)
        
        # Detect court keypoints for homography
        kp_result = keypoint_model.infer(frame, confidence=KEYPOINT_CONFIDENCE)[0]
        key_points = sv.KeyPoints.from_inference(kp_result)
        landmarks_mask = key_points.confidence[0] > KEYPOINT_ANCHOR_CONFIDENCE
        
        # Draw keypoints on frame
        for i, (pt, c) in enumerate(zip(key_points.xy[0], key_points.confidence[0])):
            if c > KEYPOINT_CONFIDENCE:
                color = (0, 255, 0) if c > KEYPOINT_ANCHOR_CONFIDENCE else (0, 165, 255)
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 5, color, -1)
        
        # Create tactical court view
        court = draw_court(config=config)
        
        # Make court blue
        court_hsv = cv2.cvtColor(court, cv2.COLOR_BGR2HSV)
        court_hsv[:,:,0] = 110
        court_hsv[:,:,1] = np.clip(court_hsv[:,:,1] * 1.5, 0, 255).astype(np.uint8)
        court = cv2.cvtColor(court_hsv, cv2.COLOR_HSV2BGR)
        
        if np.count_nonzero(landmarks_mask) >= 4 and len(players) > 0:
            # Calculate homography
            court_landmarks = np.array(config.vertices)[landmarks_mask]
            frame_landmarks = key_points[:, landmarks_mask].xy[0]
            
            transformer = ViewTransformer(
                source=frame_landmarks,
                target=court_landmarks
            )
            
            # Get player feet positions
            frame_xy = players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            
            # Transform to court coordinates
            court_xy = transformer.transform_points(points=frame_xy)
            
            # Draw players on court by team
            if len(teams) > 0:
                court = draw_points_on_court(
                    config=config,
                    xy=court_xy[teams == 0],
                    fill_color=sv.Color.from_hex(TEAM_COLORS[TEAM_NAMES[0]]),
                    court=court
                )
                court = draw_points_on_court(
                    config=config,
                    xy=court_xy[teams == 1],
                    fill_color=sv.Color.from_hex(TEAM_COLORS[TEAM_NAMES[1]]),
                    court=court
                )
        
        # Overlay tactical court on frame
        court_h, court_w = court.shape[:2]
        scale = 0.35
        new_w = int(width * scale)
        new_h = int(court_h * (new_w / court_w))
        court_resized = cv2.resize(court, (new_w, new_h))
        
        x = width - new_w - 15
        y = height - new_h - 15
        
        # Background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x-10, y-10), (x+new_w+10, y+new_h+10), (20, 20, 20), -1)
        annotated = cv2.addWeighted(overlay, 0.9, annotated, 0.1, 0)
        cv2.rectangle(annotated, (x-10, y-10), (x+new_w+10, y+new_h+10), (80, 80, 80), 2)
        
        annotated[y:y+new_h, x:x+new_w] = court_resized
        cv2.putText(annotated, "TACTICAL VIEW", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Save frame
        output_path = os.path.join(OUTPUT_DIR, f"frame_{idx+1:02d}.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"   Saved: {output_path}")
        
        # Also save tactical-only
        tactical_path = os.path.join(OUTPUT_DIR, f"tactical_{idx+1:02d}.jpg")
        cv2.imwrite(tactical_path, court)
    
    print("\n[6/7] Saving combined court view...")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "tactical_only.jpg"), court)
    
    print("\n" + "=" * 60)
    print(f"Done! Output in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
