"""
Test the full tactical view pipeline with team classification and smoothing.

This script:
1. Detects players and referees
2. Classifies players into teams using jersey colors
3. Transforms positions to court coordinates via homography
4. Applies path smoothing for stable output
5. Generates a tactical view video
"""
import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv(override=True)

from player_referee_detector import PlayerRefereeDetector, PLAYER_CLASS_IDS, REFEREE_CLASS_IDS
from team_classifier import TeamClassifier, get_player_crops
from tactical_view import TacticalView, create_combined_view, get_positions_from_detections
from path_smoothing import smooth_tactical_positions

# Constants
WARMUP_FRAMES = 30  # Frames to collect for team classifier training
OUTPUT_DIR = Path("output/tactical_test")


def process_video_with_tactical_view(
    video_path: str,
    output_path: str = None,
    max_frames: int = None,
    rebuild_transformer_every: int = 30,  # Rebuild homography every N frames for stability
):
    """
    Process video with full tactical view pipeline.

    Args:
        video_path: Path to input video
        output_path: Path to output video (optional)
        max_frames: Max frames to process (optional, for testing)
        rebuild_transformer_every: How often to rebuild homography transformer
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)
    if output_path is None:
        output_path = OUTPUT_DIR / f"{video_path.stem}_tactical.mp4"

    # Initialize components
    print("Initializing components...")
    detector = PlayerRefereeDetector()
    team_classifier = TeamClassifier(n_teams=2)
    tactical_view = TacticalView()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames to process: {total_frames}")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Collect crops for team classifier during warmup
    print(f"\nPhase 1: Collecting player crops for {WARMUP_FRAMES} warmup frames...")
    warmup_crops = []
    frame_idx = 0

    while frame_idx < WARMUP_FRAMES and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_and_track(frame)

        # Filter to players only (not referees)
        if len(detections) > 0:
            player_mask = np.isin(detections.class_id, PLAYER_CLASS_IDS)
            player_detections = detections[player_mask]

            if len(player_detections) > 0:
                crops = get_player_crops(frame, player_detections)
                warmup_crops.extend(crops)

        # Build homography on first frame
        if frame_idx == 0:
            print("  Building homography transformer...")
            tactical_view.build_transformer(frame)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  Warmup frame {frame_idx}/{WARMUP_FRAMES}, crops collected: {len(warmup_crops)}")

    # Train team classifier
    print(f"\nTraining team classifier on {len(warmup_crops)} crops...")
    if len(warmup_crops) >= 2:
        team_classifier.fit(warmup_crops)
    else:
        print("  Warning: Not enough crops for team classification")

    # Reset video for main processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    detector.tracker = detector.tracker.__class__(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=1
    )

    # Process all frames
    print(f"\nPhase 2: Processing video with tactical view...")
    frame_idx = 0
    positions_history = []  # For smoothing
    team_cache = {}  # Cache team assignments by tracker_id

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and track
        detections = detector.detect_and_track(frame)

        # Rebuild transformer periodically for stability
        if frame_idx % rebuild_transformer_every == 0:
            tactical_view.build_transformer(frame)

        # Get team assignments
        team_assignments = {}
        if len(detections) > 0 and detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                tracker_id = int(tracker_id)
                class_id = detections.class_id[i]

                # Referees get special team assignment
                if class_id in REFEREE_CLASS_IDS:
                    team_assignments[tracker_id] = -1
                elif tracker_id in team_cache:
                    # Use cached team assignment
                    team_assignments[tracker_id] = team_cache[tracker_id]
                else:
                    # Classify this player
                    if team_classifier.is_fitted:
                        player_det = detections[np.array([i])]
                        crops = get_player_crops(frame, player_det)
                        if crops:
                            team_id = team_classifier.predict(crops)[0]
                            team_assignments[tracker_id] = team_id
                            team_cache[tracker_id] = team_id
                        else:
                            team_assignments[tracker_id] = 0
                    else:
                        team_assignments[tracker_id] = 0

        # Get positions for smoothing history
        positions = get_positions_from_detections(detections)
        positions_history.append(positions)

        # Keep only recent history for smoothing
        if len(positions_history) > 15:
            positions_history = positions_history[-15:]

        # Render tactical view
        tactical_img = tactical_view.render_from_detections(
            detections,
            team_assignments=team_assignments,
            team_colors=team_classifier.team_colors
        )

        # Annotate main frame with detections
        annotated = detector.annotate_frame(frame, detections)

        # Add team colors to annotations
        if len(detections) > 0 and detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                tracker_id = int(tracker_id)
                x1, y1, x2, y2 = map(int, detections.xyxy[i])

                team_id = team_assignments.get(tracker_id, 0)
                color = team_classifier.get_team_color(team_id)

                # Draw small colored dot for team
                cx = (x1 + x2) // 2
                cv2.circle(annotated, (cx, y1 - 5), 8, color, -1)

        # Combine views
        combined = create_combined_view(annotated, tactical_img)

        # Write frame
        writer.write(combined)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    print(f"\nDone! Output saved to: {output_path}")
    print(f"Team assignments cached for {len(team_cache)} unique players")

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test tactical view pipeline")
    parser.add_argument("video_path", nargs="?",
                        default="../test_videos/test_game.mp4",
                        help="Path to input video")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (for testing)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video path")

    args = parser.parse_args()

    process_video_with_tactical_view(
        args.video_path,
        output_path=args.output,
        max_frames=args.max_frames
    )
