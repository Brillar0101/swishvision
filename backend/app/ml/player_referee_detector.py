"""
Player and Referee Detection using fine-tuned RF-DETR model.

This module detects players and referees using a locally trained RF-DETR model,
eliminating the need for Roboflow API calls during inference.

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
from pathlib import Path

# Model paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
RFDETR_CHECKPOINT = MODEL_DIR / "rfdetr_basketball.pth"

# Detection settings
DETECTION_CONFIDENCE = 0.4
DETECTION_IOU_THRESHOLD = 0.5

# Player class IDs (all player variants)
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]  # player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block

# Referee class ID
REFEREE_CLASS_IDS = [8]

# All classes we want to detect
TARGET_CLASS_IDS = PLAYER_CLASS_IDS + REFEREE_CLASS_IDS

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
    Detects players and referees using fine-tuned RF-DETR model.
    Designed to feed bounding boxes to SAM2 for segmentation.
    """

    def __init__(self, confidence: float = DETECTION_CONFIDENCE,
                 iou_threshold: float = DETECTION_IOU_THRESHOLD,
                 checkpoint_path: str = None):
        """
        Initialize the detector.

        Args:
            confidence: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            checkpoint_path: Path to RF-DETR checkpoint (uses default if None)
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = None
        self._use_rfdetr = False

        # Determine which checkpoint to use
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = RFDETR_CHECKPOINT

        # Try to load RF-DETR model
        self._load_model()

        # Tracker for persistent IDs
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.15,  # Lower threshold to pick up re-entering players
            lost_track_buffer=600,  # Keep tracking for 20 sec if lost (allows players to exit/re-enter)
            minimum_matching_threshold=0.5,  # Lower threshold for better re-association
            frame_rate=30,
            minimum_consecutive_frames=1
        )

        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    def _load_model(self):
        """Load Roboflow API model (as per reference notebook implementation)."""
        from dotenv import load_dotenv
        from inference import get_model

        load_dotenv(override=True)

        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is required")

        self.model = get_model(
            model_id="basketball-player-detection-3-ycjdo/4",
            api_key=api_key
        )
        self._use_rfdetr = False
        print("Using Roboflow API for detection")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect players and referees in a frame.

        Returns:
            sv.Detections with only player and referee detections
        """
        if self._use_rfdetr:
            # RF-DETR inference - returns sv.Detections directly
            detections = self.model.predict(frame, threshold=self.confidence)
        else:
            # Roboflow API inference
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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to RF-DETR checkpoint")

    args = parser.parse_args()

    results = process_video(args.video_path, args.output_dir)
    print(f"\nOutput video: {results['output_video']}")
