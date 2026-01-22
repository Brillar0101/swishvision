import os
import numpy as np
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
from pathlib import Path

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


def process_video(source_video_path: str, output_dir: str = None) -> dict:
    """
    Process video with basketball object detection using sv.process_video.

    Args:
        source_video_path: Path to input video
        output_dir: Output directory (optional, defaults to same directory as source)

    Returns:
        dict with output paths and stats
    """
    source_path = Path(source_video_path)

    # Set up output paths
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        target_video_path = Path(output_dir) / f"{source_path.stem}-detection.mp4"
    else:
        target_video_path = source_path.parent / f"{source_path.stem}-detection.mp4"

    # Load model
    model = get_model(
        model_id=PLAYER_DETECTION_MODEL_ID,
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )
    print(f"Model {PLAYER_DETECTION_MODEL_ID} loaded successfully.")

    # Set up annotators
    box_annotator = sv.BoxAnnotator(color=COLOR, thickness=2)
    label_annotator = sv.LabelAnnotator(color=COLOR, text_color=sv.Color.BLACK)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        # Run inference
        result = model.infer(
            frame,
            confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
            iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
        )[0]
        detections = sv.Detections.from_inference(result)

        # Filter out class 2 (number)
        if len(detections) > 0:
            mask = detections.class_id != 2
            detections = detections[mask]

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        return annotated_frame

    # Process video
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

    parser = argparse.ArgumentParser(description="Basketball Object Detection using ycjdo/4 model")
    parser.add_argument("video_path", nargs="?", default="../test_videos/test_game.mp4",
                        help="Path to input video")
    parser.add_argument("output_dir", nargs="?", default="output/basketball_detection",
                        help="Output directory")

    args = parser.parse_args()

    results = process_video(args.video_path, args.output_dir)
    print(f"\nOutput video: {results['output_video']}")
