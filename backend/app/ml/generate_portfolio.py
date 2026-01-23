"""
Generate portfolio demonstration videos for SwishVision.

This script generates separate videos showing each stage of the pipeline:
1. Player Detection - RF-DETR bounding boxes
2. Team Classification - Color-coded team assignments
3. Player Tracking - Persistent IDs with trails
4. Tactical View - 2D court minimap

Usage:
    python generate_portfolio.py video.mp4 --max-frames 300
"""
import os
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from collections import defaultdict

from player_referee_detector import PlayerRefereeDetector, PLAYER_CLASS_IDS, REFEREE_CLASS_IDS, CLASS_NAMES
from team_classifier import TeamClassifier, get_player_crops
from tactical_view import TacticalView, create_combined_view, get_positions_from_detections


def generate_portfolio_videos(
    video_path: str,
    output_dir: str = None,
    max_frames: int = None,
    warmup_frames: int = 30,
):
    """
    Generate portfolio videos showing each pipeline stage.

    Args:
        video_path: Path to input video
        output_dir: Output directory
        max_frames: Max frames to process
        warmup_frames: Frames for team classifier training
    """
    video_path = Path(video_path)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "training" / "outputs" / "portfolio"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # First pass: collect crops for team classifier
    print(f"\nPhase 1: Collecting crops for team classification ({warmup_frames} frames)...")
    warmup_crops = []

    for i in range(min(warmup_frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        if len(detections) > 0:
            player_mask = np.isin(detections.class_id, PLAYER_CLASS_IDS)
            player_dets = detections[player_mask]
            if len(player_dets) > 0:
                crops = get_player_crops(frame, player_dets)
                warmup_crops.extend(crops)

        # Build homography on first frame
        if i == 0:
            tactical_view.build_transformer(frame)

    print(f"  Collected {len(warmup_crops)} crops")

    # Train team classifier
    if len(warmup_crops) >= 2:
        team_classifier.fit(warmup_crops)
        print("  Team classifier trained")

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    detector.tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=1
    )

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = {
        'detection': cv2.VideoWriter(str(output_dir / "stage1_detection.mp4"), fourcc, fps, (width, height)),
        'tracking': cv2.VideoWriter(str(output_dir / "stage2_tracking.mp4"), fourcc, fps, (width, height)),
        'teams': cv2.VideoWriter(str(output_dir / "stage3_teams.mp4"), fourcc, fps, (width, height)),
        'tactical': cv2.VideoWriter(str(output_dir / "stage4_tactical.mp4"), fourcc, fps, (width, height)),
        'combined': cv2.VideoWriter(str(output_dir / "stage5_combined.mp4"), fourcc, fps, (width, height)),
    }

    # Track history for trails
    track_history = defaultdict(list)
    team_cache = {}

    # Color palette for tracking
    track_colors = sv.ColorPalette.from_hex([
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    ])

    print(f"\nPhase 2: Generating portfolio videos...")

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and track
        detections = detector.detect_and_track(frame)

        # Rebuild homography periodically
        if frame_idx % 30 == 0:
            tactical_view.build_transformer(frame)

        # Get team assignments
        team_assignments = {}
        if len(detections) > 0 and detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                tracker_id = int(tracker_id)
                class_id = detections.class_id[i]

                if class_id in REFEREE_CLASS_IDS:
                    team_assignments[tracker_id] = -1
                elif tracker_id in team_cache:
                    team_assignments[tracker_id] = team_cache[tracker_id]
                elif team_classifier.is_fitted:
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

        # ========== Stage 1: Detection ==========
        detection_frame = frame.copy()
        add_stage_label(detection_frame, "Stage 1: Player Detection (RF-DETR)")

        if len(detections) > 0:
            box_annotator = sv.BoxAnnotator(thickness=2)
            detection_frame = box_annotator.annotate(detection_frame, detections)

            # Add count
            cv2.putText(detection_frame, f"Detected: {len(detections)}", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(detection_frame, f"Detected: {len(detections)}", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writers['detection'].write(detection_frame)

        # ========== Stage 2: Tracking ==========
        tracking_frame = frame.copy()
        add_stage_label(tracking_frame, "Stage 2: Player Tracking (ByteTrack)")

        if len(detections) > 0 and detections.tracker_id is not None:
            for i in range(len(detections)):
                tracker_id = int(detections.tracker_id[i])
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                cx, cy = (x1 + x2) // 2, y2

                # Update trail
                track_history[tracker_id].append((cx, cy))
                if len(track_history[tracker_id]) > 30:
                    track_history[tracker_id] = track_history[tracker_id][-30:]

                # Get color
                color_idx = tracker_id % len(track_colors.colors)
                color = track_colors.colors[color_idx]
                bgr = (int(color.b), int(color.g), int(color.r))

                # Draw trail
                points = track_history[tracker_id]
                for j in range(1, len(points)):
                    thickness = max(1, int(3 * j / len(points)))
                    cv2.line(tracking_frame, points[j-1], points[j], bgr, thickness)

                # Draw box and label
                cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), bgr, 2)
                label = f"#{tracker_id}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(tracking_frame, (x1, y1 - lh - 10), (x1 + lw + 5, y1), bgr, -1)
                cv2.putText(tracking_frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writers['tracking'].write(tracking_frame)

        # ========== Stage 3: Team Classification ==========
        teams_frame = frame.copy()
        add_stage_label(teams_frame, "Stage 3: Team Classification (SigLIP + K-means)")

        if len(detections) > 0 and detections.tracker_id is not None:
            team_counts = {0: 0, 1: 0, -1: 0}

            for i in range(len(detections)):
                tracker_id = int(detections.tracker_id[i])
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                team_id = team_assignments.get(tracker_id, 0)
                team_counts[team_id] = team_counts.get(team_id, 0) + 1

                color = team_classifier.get_team_color(team_id)

                # Draw box
                cv2.rectangle(teams_frame, (x1, y1), (x2, y2), color, 2)

                # Draw team indicator dot
                cx = (x1 + x2) // 2
                cv2.circle(teams_frame, (cx, y1 - 15), 12, color, -1)
                cv2.circle(teams_frame, (cx, y1 - 15), 12, (255, 255, 255), 2)

            # Team counts
            cv2.putText(teams_frame, f"Team A: {team_counts[0]}  Team B: {team_counts[1]}  Refs: {team_counts[-1]}",
                       (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(teams_frame, f"Team A: {team_counts[0]}  Team B: {team_counts[1]}  Refs: {team_counts[-1]}",
                       (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writers['teams'].write(teams_frame)

        # ========== Stage 4: Tactical View ==========
        tactical_frame = frame.copy()
        add_stage_label(tactical_frame, "Stage 4: Tactical View (Homography)")

        positions = get_positions_from_detections(detections)
        tactical_img = tactical_view.render(
            player_positions=positions,
            frame_shape=(height, width),
            team_assignments=team_assignments,
            team_colors=team_classifier.team_colors
        )

        # Overlay tactical view
        tactical_h, tactical_w = tactical_img.shape[:2]
        x_offset = width - tactical_w - 20
        y_offset = height - tactical_h - 20
        tactical_frame[y_offset:y_offset+tactical_h, x_offset:x_offset+tactical_w] = tactical_img

        writers['tactical'].write(tactical_frame)

        # ========== Stage 5: Combined ==========
        combined_frame = frame.copy()

        # Draw detections with team colors
        if len(detections) > 0 and detections.tracker_id is not None:
            for i in range(len(detections)):
                tracker_id = int(detections.tracker_id[i])
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                team_id = team_assignments.get(tracker_id, 0)
                color = team_classifier.get_team_color(team_id)

                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, 2)

                # Label
                class_id = detections.class_id[i]
                class_name = CLASS_NAMES.get(class_id, "player")
                label = f"#{tracker_id}"
                cv2.putText(combined_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add tactical overlay
        combined_frame[y_offset:y_offset+tactical_h, x_offset:x_offset+tactical_w] = tactical_img

        add_stage_label(combined_frame, "SwishVision - Full Pipeline")

        writers['combined'].write(combined_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")

    # Cleanup
    cap.release()
    for writer in writers.values():
        writer.release()

    print(f"\nPortfolio videos saved to: {output_dir}")
    print("  - stage1_detection.mp4")
    print("  - stage2_tracking.mp4")
    print("  - stage3_teams.mp4")
    print("  - stage4_tactical.mp4")
    print("  - stage5_combined.mp4")

    return {name: str(output_dir / f"stage{i+1}_{name}.mp4") for i, name in enumerate(['detection', 'tracking', 'teams', 'tactical', 'combined'])}


def add_stage_label(frame, text):
    """Add stage label to top of frame."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate portfolio videos")
    parser.add_argument("video_path", nargs="?", default="../../test_videos/test_game.mp4",
                        help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process")

    args = parser.parse_args()

    generate_portfolio_videos(
        args.video_path,
        output_dir=args.output,
        max_frames=args.max_frames,
    )
