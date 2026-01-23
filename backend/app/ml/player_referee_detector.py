"""
Player and Referee Detection using basketball-player-detection-3-ycjdo/4 model.

This module detects only players and referees (classes 3, 4, 5, 6, 7, 8)
for use with SAM2 segmentation pipeline.

Player classes:
- 3: player
- 4: player-in-possession
- 5: player-jump-shot
- 6: player-layup-dunk
- 7: player-shot-block

Referee class:
- 8: referee
"""

import os
import numpy as np
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

# Model configuration
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.4
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9

# Player class IDs (all player variants)
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]  # player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block

# Referee class ID
REFEREE_CLASS_IDS = [8]

# All classes we want to detect
TARGET_CLASS_IDS = PLAYER_CLASS_IDS + REFEREE_CLASS_IDS

# Color palette - blue for players, cyan for referees
COLOR = sv.ColorPalette.from_hex([
    "#3399ff",  # Blue - player
    "#66ffff",  # Cyan - referee
])

# Class names
CLASS_NAMES = {
    3: "player",
    4: "player",  # player-in-possession -> player
    5: "player",  # player-jump-shot -> player
    6: "player",  # player-layup-dunk -> player
    7: "player",  # player-shot-block -> player
    8: "referee",
}


class PlayerRefereeDetector:
    """
    Detects players and referees using the ycjdo/4 basketball model.
    Designed to feed bounding boxes to SAM2 for segmentation.
    """

    def __init__(self, confidence: float = PLAYER_DETECTION_MODEL_CONFIDENCE,
                 iou_threshold: float = PLAYER_DETECTION_MODEL_IOU_THRESHOLD):
        self.model = get_model(
            model_id=PLAYER_DETECTION_MODEL_ID,
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Tracker for persistent IDs
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,  # Keep tracking for 2 sec if lost
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=1
        )

        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect players and referees in a frame.

        Returns:
            sv.Detections with only player and referee detections
        """
        result = self.model.infer(
            frame,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold
        )[0]
        detections = sv.Detections.from_inference(result)

        # Filter to only players and referees
        if len(detections) > 0:
            mask = np.isin(detections.class_id, TARGET_CLASS_IDS)
            detections = detections[mask]

        return detections

    def detect_and_track(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect and track players/referees with persistent IDs.

        Returns:
            sv.Detections with tracker_id assigned
        """
        detections = self.detect(frame)

        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)

        return detections

    def get_boxes_for_sam2(self, detections: sv.Detections) -> list:
        """
        Convert detections to box format for SAM2.

        Returns:
            List of dicts with 'box', 'class', 'tracker_id'
        """
        boxes = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            class_id = detections.class_id[i]
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else i

            boxes.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'class': 'referee' if class_id == 8 else 'player',
                'tracker_id': int(tracker_id),
                'confidence': float(detections.confidence[i])
            })

        return boxes

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw detections on frame."""
        labels = []
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            class_name = CLASS_NAMES.get(class_id, "unknown")
            conf = detections.confidence[i]

            if detections.tracker_id is not None:
                label = f"#{detections.tracker_id[i]} {class_name} {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"
            labels.append(label)

        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        return annotated


def process_video(source_video_path: str, output_dir: str = None) -> dict:
    """
    Process video detecting only players and referees.

    Args:
        source_video_path: Path to input video
        output_dir: Output directory

    Returns:
        dict with output paths
    """
    source_path = Path(source_video_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        target_video_path = Path(output_dir) / f"{source_path.stem}-players-referees.mp4"
    else:
        target_video_path = source_path.parent / f"{source_path.stem}-players-referees.mp4"

    detector = PlayerRefereeDetector()
    print(f"Model {PLAYER_DETECTION_MODEL_ID} loaded successfully.")
    print(f"Detecting: players (classes 3-7) and referees (class 8)")

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        detections = detector.detect_and_track(frame)
        return detector.annotate_frame(frame, detections)

    print(f"Processing video: {source_video_path}")
    print(f"Output: {target_video_path}")

    sv.process_video(
        source_path=str(source_path),
        target_path=str(target_video_path),
        callback=callback,
        show_progress=True
    )

    print(f"\n=== Processing Complete ===")
    print(f"Output video: {target_video_path}")

    return {
        "source_video": str(source_path),
        "output_video": str(target_video_path)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Player and Referee Detection")
    parser.add_argument("video_path", nargs="?", default="../test_videos/test_game.mp4",
                        help="Path to input video")
    parser.add_argument("output_dir", nargs="?", default="output/player_referee_detection",
                        help="Output directory")

    args = parser.parse_args()

    results = process_video(args.video_path, args.output_dir)
    print(f"\nOutput video: {results['output_video']}")
