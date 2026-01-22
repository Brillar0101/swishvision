import os
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Model configuration
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.4
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9

# Color palette for different classes
COLOR = sv.ColorPalette.from_hex([
    "#ffff00",  # Yellow - ball
    "#ff9b00",  # Orange - ball-in-basket
    "#ff66ff",  # Pink - number
    "#3399ff",  # Blue - player
    "#ff66b2",  # Light pink - player-in-possession
    "#ff8080",  # Light red - player-jump-shot
    "#b266ff",  # Purple - player-layup-dunk
    "#9999ff",  # Light purple - player-shot-block
    "#66ffff",  # Cyan - referee
    "#33ff99",  # Green - rim
    "#66ff66",  # Light green - extra
    "#99ff00"   # Yellow-green - extra
])

# Class names from the basketball-player-detection-3-ycjdo/4 model
CLASS_NAMES = {
    0: "ball",
    1: "ball-in-basket",
    2: "number",
    3: "player",
    4: "player-in-possession",
    5: "player-jump-shot",
    6: "player-layup-dunk",
    7: "player-shot-block",
    8: "referee",
    9: "rim"
}


class BasketballObjectDetector:
    """
    Uses the basketball-player-detection-3-ycjdo/4 model to detect all basketball
    game objects including: ball, ball-in-basket, number, player, player-in-possession,
    player-jump-shot, player-layup-dunk, player-shot-block, referee, and rim.
    """

    def __init__(self, confidence: float = PLAYER_DETECTION_MODEL_CONFIDENCE,
                 iou_threshold: float = PLAYER_DETECTION_MODEL_IOU_THRESHOLD):
        # Load the Roboflow basketball detection model
        self.model = get_model(
            model_id=PLAYER_DETECTION_MODEL_ID,
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Annotators
        self.box_annotator = sv.BoxAnnotator(color=COLOR, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=COLOR,
            text_color=sv.Color.BLACK,
            text_scale=0.5,
            text_thickness=1
        )

        # Tracker for persistent IDs
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=1
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on a frame and return all detected objects."""
        result = self.model.infer(
            frame,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold
        )[0]
        detections = sv.Detections.from_inference(result)

        # Filter out class 2 (number) - not needed for tracking
        if len(detections) > 0:
            mask = detections.class_id != 2
            detections = detections[mask]

        return detections

    def get_labels(self, detections: sv.Detections) -> list:
        """Generate labels for each detection with class name and confidence."""
        labels = []
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

            # Add tracker ID if available
            if detections.tracker_id is not None and detections.tracker_id[i] is not None:
                label = f"#{detections.tracker_id[i]} {class_name} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"
            labels.append(label)
        return labels

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw all detections on the frame with labels."""
        labels = self.get_labels(detections)

        annotated = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels
        )
        return annotated

    def count_by_class(self, detections: sv.Detections) -> dict:
        """Count detections by class."""
        counts = {}
        for class_id in detections.class_id:
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def process_video(self, video_path: str, output_dir: str,
                      max_seconds: float = None,
                      use_tracking: bool = True) -> dict:
        """
        Process entire video with basketball object detection.

        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            max_seconds: Maximum seconds to process (None for full video)
            use_tracking: Whether to use ByteTrack for persistent IDs
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Limit to max_seconds if specified
        if max_seconds:
            max_frames = int(fps * max_seconds)
            total_frames = min(total_frames, max_frames)
            print(f"Processing {total_frames} frames ({max_seconds}s at {fps:.1f} FPS)")
        else:
            duration = total_frames / fps
            print(f"Processing full video: {total_frames} frames ({duration:.1f}s at {fps:.1f} FPS)")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_dir, "basketball_detection.mp4")
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Stats tracking
        stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "class_counts": {}
        }

        for frame_idx in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            detections = self.detect(frame)

            # Apply tracking for persistent IDs
            if use_tracking and len(detections) > 0:
                detections = self.tracker.update_with_detections(detections)

            # Update stats
            stats["frames_processed"] += 1
            stats["total_detections"] += len(detections)

            # Count by class
            frame_counts = self.count_by_class(detections)
            for class_name, count in frame_counts.items():
                stats["class_counts"][class_name] = stats["class_counts"].get(class_name, 0) + count

            # Annotate frame
            annotated = self.annotate_frame(frame, detections)

            # Add frame info overlay
            cv2.putText(annotated, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Detections: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show class breakdown
            y_offset = 90
            for class_name, count in sorted(frame_counts.items()):
                cv2.putText(annotated, f"{class_name}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 20

            output_video.write(annotated)

        cap.release()
        output_video.release()

        # Print summary
        print(f"\n=== Processing Complete ===")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average detections/frame: {stats['total_detections'] / max(stats['frames_processed'], 1):.1f}")
        print(f"\nDetections by class (total across all frames):")
        for class_name, count in sorted(stats["class_counts"].items()):
            avg = count / stats["frames_processed"]
            print(f"  {class_name}: {count} (avg {avg:.1f}/frame)")

        return {
            "video_path": video_path,
            "output_dir": output_dir,
            "output_video": output_path,
            "stats": stats
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Basketball Object Detection using ycjdo/4 model")
    parser.add_argument("video_path", nargs="?", default="../test_videos/test_game.mp4",
                        help="Path to input video")
    parser.add_argument("output_dir", nargs="?", default="output/basketball_detection",
                        help="Output directory")
    parser.add_argument("--confidence", type=float, default=PLAYER_DETECTION_MODEL_CONFIDENCE,
                        help=f"Detection confidence threshold (default: {PLAYER_DETECTION_MODEL_CONFIDENCE})")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Max seconds of video to process (default: full video)")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable object tracking")

    args = parser.parse_args()

    detector = BasketballObjectDetector(confidence=args.confidence)

    results = detector.process_video(
        args.video_path,
        args.output_dir,
        max_seconds=args.max_seconds,
        use_tracking=not args.no_tracking
    )

    print(f"\nOutput video: {results['output_video']}")
