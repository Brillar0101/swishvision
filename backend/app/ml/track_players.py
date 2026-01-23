"""
Player Tracking Demo - Track all players with persistent IDs.

This script demonstrates player tracking using ByteTrack:
- Detects players and referees using RF-DETR
- Assigns persistent tracker IDs across frames
- Handles players entering/exiting the frame
- Outputs annotated video with tracking visualization
"""
import os
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from collections import defaultdict

from player_referee_detector import PlayerRefereeDetector, PLAYER_CLASS_IDS, REFEREE_CLASS_IDS, CLASS_NAMES


def track_players(
    video_path: str,
    output_path: str = None,
    max_frames: int = None,
    show_trails: bool = True,
    trail_length: int = 30,
):
    """
    Track players throughout a video with persistent IDs.

    Args:
        video_path: Path to input video
        output_path: Path to output video (optional)
        max_frames: Max frames to process (optional)
        show_trails: Whether to show movement trails
        trail_length: Number of frames to keep in trail history
    """
    video_path = Path(video_path)

    if output_path is None:
        output_dir = Path(__file__).parent.parent.parent / "training" / "outputs" / "tracking"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_tracked.mp4"

    # Initialize detector
    print("Initializing player detector...")
    detector = PlayerRefereeDetector()

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

    # Track history for trails
    track_history = defaultdict(list)

    # Colors for different tracker IDs
    colors = sv.ColorPalette.from_hex([
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FFD700",
    ])

    # Stats tracking
    active_trackers = set()
    all_trackers = set()
    entries = []  # (frame_idx, tracker_id)
    exits = []    # (frame_idx, tracker_id)

    frame_idx = 0
    print(f"\nProcessing video...")

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and track
        detections = detector.detect_and_track(frame)

        # Get current tracker IDs
        current_trackers = set()
        if len(detections) > 0 and detections.tracker_id is not None:
            current_trackers = set(int(tid) for tid in detections.tracker_id)

        # Detect entries (new trackers)
        new_trackers = current_trackers - active_trackers
        for tid in new_trackers:
            entries.append((frame_idx, tid))
            all_trackers.add(tid)
            if frame_idx > 0:  # Don't announce on first frame
                print(f"  Frame {frame_idx}: Player #{tid} entered")

        # Detect exits (lost trackers)
        lost_trackers = active_trackers - current_trackers
        for tid in lost_trackers:
            exits.append((frame_idx, tid))
            if frame_idx < total_frames - 1:  # Don't announce on last frame
                print(f"  Frame {frame_idx}: Player #{tid} exited")

        active_trackers = current_trackers

        # Draw on frame
        annotated = frame.copy()

        # Update track history and draw trails
        if len(detections) > 0 and detections.tracker_id is not None:
            for i in range(len(detections)):
                tracker_id = int(detections.tracker_id[i])
                x1, y1, x2, y2 = map(int, detections.xyxy[i])

                # Calculate center point (bottom center for feet position)
                cx = (x1 + x2) // 2
                cy = y2  # Bottom of bounding box

                # Add to history
                track_history[tracker_id].append((cx, cy))

                # Keep only recent history
                if len(track_history[tracker_id]) > trail_length:
                    track_history[tracker_id] = track_history[tracker_id][-trail_length:]

                # Get color for this tracker
                color_idx = tracker_id % len(colors.colors)
                color = colors.colors[color_idx]
                bgr_color = (int(color.b), int(color.g), int(color.r))

                # Draw trail
                if show_trails and len(track_history[tracker_id]) > 1:
                    points = track_history[tracker_id]
                    for j in range(1, len(points)):
                        # Fade trail based on age
                        alpha = j / len(points)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(annotated, points[j-1], points[j], bgr_color, thickness)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 2)

                # Draw label
                class_id = detections.class_id[i]
                class_name = CLASS_NAMES.get(class_id, "unknown")
                conf = detections.confidence[i]
                label = f"#{tracker_id} {class_name}"

                # Label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), bgr_color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw center dot
                cv2.circle(annotated, (cx, cy), 5, bgr_color, -1)

        # Draw stats overlay
        stats_text = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Active Players: {len(active_trackers)}",
            f"Total Tracked: {len(all_trackers)}",
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(annotated, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(annotated, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    # Print summary
    print(f"\n{'='*60}")
    print("Tracking Summary")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Unique players tracked: {len(all_trackers)}")
    print(f"Player entries: {len(entries)}")
    print(f"Player exits: {len(exits)}")
    print(f"\nOutput saved to: {output_path}")

    return {
        "output_video": str(output_path),
        "total_frames": frame_idx,
        "unique_players": len(all_trackers),
        "tracker_ids": list(all_trackers),
        "entries": entries,
        "exits": exits,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track players in basketball video")
    parser.add_argument("video_path", nargs="?",
                        default="../../test_videos/test_game.mp4",
                        help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process")
    parser.add_argument("--no-trails", action="store_true",
                        help="Disable movement trails")
    parser.add_argument("--trail-length", type=int, default=30,
                        help="Trail length in frames")

    args = parser.parse_args()

    track_players(
        args.video_path,
        output_path=args.output,
        max_frames=args.max_frames,
        show_trails=not args.no_trails,
        trail_length=args.trail_length,
    )
