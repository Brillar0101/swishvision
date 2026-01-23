"""
Test homography transformation with image outputs.
Generates images showing:
1. Original frame with detections
2. Tactical court view with player positions
3. Combined view
"""
import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

from player_referee_detector import PlayerRefereeDetector, PLAYER_CLASS_IDS, REFEREE_CLASS_IDS
from team_classifier import TeamClassifier, get_player_crops
from tactical_view import TacticalView, create_combined_view, get_positions_from_detections

OUTPUT_DIR = Path("output/homography_test")


def test_homography_on_video(video_path: str, frame_indices: list = None):
    """
    Test homography on specific frames and save images.

    Args:
        video_path: Path to video
        frame_indices: List of frame indices to test (default: [0, 50, 100, 150, 200])
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if frame_indices is None:
        frame_indices = [0, 50, 100, 150, 200]

    # Initialize
    print("Loading models...")
    detector = PlayerRefereeDetector()
    team_classifier = TeamClassifier(n_teams=2)
    tactical_view = TacticalView()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")

    # Filter valid frame indices
    frame_indices = [i for i in frame_indices if i < total_frames]

    # Collect crops for team classifier from first 30 frames
    print("\nCollecting crops for team classification...")
    all_crops = []
    for i in range(min(30, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        if len(detections) > 0:
            player_mask = np.isin(detections.class_id, PLAYER_CLASS_IDS)
            player_detections = detections[player_mask]
            if len(player_detections) > 0:
                crops = get_player_crops(frame, player_detections)
                all_crops.extend(crops)

    print(f"Collected {len(all_crops)} crops")
    if len(all_crops) >= 2:
        team_classifier.fit(all_crops)

    # Process each target frame
    print(f"\nProcessing frames: {frame_indices}")

    for frame_idx in frame_indices:
        print(f"\n--- Frame {frame_idx} ---")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Could not read frame {frame_idx}")
            continue

        # Detect players (use detect_and_track to get tracker_id)
        detections = detector.detect_and_track(frame)
        print(f"  Detected {len(detections)} objects")

        if len(detections) == 0:
            print("  No detections, skipping...")
            continue

        # Build homography transformer
        transformer = tactical_view.build_transformer(frame)
        if transformer is None:
            print("  Could not build homography transformer")
        else:
            print("  Homography transformer built successfully")

        # Get team assignments using tracker_id as key
        team_assignments = {}
        if detections.tracker_id is not None:
            for i in range(len(detections)):
                tracker_id = int(detections.tracker_id[i])
                class_id = detections.class_id[i]

                if class_id in REFEREE_CLASS_IDS:
                    team_assignments[tracker_id] = -1  # Referee
                elif team_classifier.is_fitted:
                    det_single = detections[np.array([i])]
                    crops = get_player_crops(frame, det_single)
                    if crops:
                        team_id = team_classifier.predict(crops)[0]
                        team_assignments[tracker_id] = team_id
                    else:
                        team_assignments[tracker_id] = 0
                else:
                    team_assignments[tracker_id] = 0

        # Count teams
        team_counts = {-1: 0, 0: 0, 1: 0}
        for t in team_assignments.values():
            team_counts[t] = team_counts.get(t, 0) + 1
        print(f"  Teams: Team A={team_counts[0]}, Team B={team_counts[1]}, Refs={team_counts[-1]}")

        # Get positions (keyed by tracker_id)
        positions = get_positions_from_detections(detections)
        print(f"  Positions extracted: {len(positions)}")

        # Render tactical view
        tactical_img = tactical_view.render(
            player_positions=positions,
            frame_shape=(frame.shape[0], frame.shape[1]),
            team_assignments=team_assignments,
            team_colors=team_classifier.team_colors
        )

        # Annotate original frame
        annotated = detector.annotate_frame(frame, detections)

        # Add team color dots
        if detections.tracker_id is not None:
            for i in range(len(detections)):
                tracker_id = int(detections.tracker_id[i])
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                team_id = team_assignments.get(tracker_id, 0)
                color = team_classifier.get_team_color(team_id)
                cx = (x1 + x2) // 2
                cv2.circle(annotated, (cx, y1 - 10), 10, color, -1)
                cv2.circle(annotated, (cx, y1 - 10), 10, (255, 255, 255), 2)

        # Create combined view
        combined = create_combined_view(annotated, tactical_img)

        # Save images
        prefix = f"frame_{frame_idx:04d}"

        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_original.jpg"), frame)
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_annotated.jpg"), annotated)
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_tactical.jpg"), tactical_img)
        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_combined.jpg"), combined)

        print(f"  Saved: {prefix}_*.jpg")

    cap.release()
    print(f"\n\nAll images saved to: {OUTPUT_DIR}")

    return str(OUTPUT_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test homography with image outputs")
    parser.add_argument("video_path", nargs="?",
                        default="../test_videos/test_game.mp4",
                        help="Path to video")
    parser.add_argument("--frames", "-f", type=int, nargs="+",
                        default=[0, 50, 100, 150, 200],
                        help="Frame indices to test")

    args = parser.parse_args()

    test_homography_on_video(args.video_path, args.frames)
