#!/usr/bin/env python3
"""
Test player detection and tracking consistency.

Tests only:
- RF-DETR player detection
- ByteTrack ID persistence
- Tracking continuity across frames

Does NOT test:
- Team classification
- Jersey numbers
- SAM2 segmentation
- Tactical view
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv(backend_path / '.env')

from app.ml.player_referee_detector import PlayerRefereeDetector
from app.ml.ui_config import Colors, put_text_pil
import supervision as sv
from tqdm import tqdm

PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]


def analyze_tracking_quality(frame_detections: Dict[int, sv.Detections]) -> Dict:
    """Analyze tracking quality metrics."""
    # Track appearance count for each ID
    tracker_appearances = defaultdict(int)
    tracker_first_frame = {}
    tracker_last_frame = {}

    for frame_idx, detections in frame_detections.items():
        if len(detections) == 0 or detections.tracker_id is None:
            continue

        for tid in detections.tracker_id:
            tid = int(tid)
            tracker_appearances[tid] += 1

            if tid not in tracker_first_frame:
                tracker_first_frame[tid] = frame_idx
            tracker_last_frame[tid] = frame_idx

    # Calculate metrics
    total_trackers = len(tracker_appearances)
    avg_appearances = np.mean(list(tracker_appearances.values())) if total_trackers > 0 else 0

    # Find persistent trackers (appear in >50% of their lifetime)
    persistent_trackers = []
    for tid, appearances in tracker_appearances.items():
        first = tracker_first_frame[tid]
        last = tracker_last_frame[tid]
        lifetime = last - first + 1
        persistence_ratio = appearances / lifetime if lifetime > 0 else 0

        if persistence_ratio > 0.8 and appearances > 30:  # Strong persistence
            persistent_trackers.append(tid)

    # Detect ID switches (many short-lived trackers)
    short_lived = sum(1 for count in tracker_appearances.values() if count < 10)

    return {
        'total_trackers': total_trackers,
        'persistent_trackers': len(persistent_trackers),
        'persistent_tracker_ids': persistent_trackers,
        'short_lived_trackers': short_lived,
        'avg_appearances': avg_appearances,
        'tracker_appearances': dict(tracker_appearances),
    }


def create_tracking_visualization(
    frames: List[np.ndarray],
    frame_detections: Dict[int, sv.Detections],
    output_path: str,
    quality_metrics: Dict,
):
    """Create video showing only detection boxes and tracker IDs."""
    if len(frames) == 0:
        print("No frames to visualize")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # Color map for tracker IDs
    np.random.seed(42)
    color_map = {}

    persistent_ids = set(quality_metrics['persistent_tracker_ids'])

    print("Creating visualization video...")
    for frame_idx in tqdm(range(len(frames))):
        frame = frames[frame_idx].copy()
        detections = frame_detections.get(frame_idx, sv.Detections.empty())

        if len(detections) > 0 and detections.tracker_id is not None:
            for i in range(len(detections)):
                tid = int(detections.tracker_id[i])
                box = detections.xyxy[i]

                # Generate consistent color for this tracker ID
                if tid not in color_map:
                    color_map[tid] = tuple(np.random.randint(100, 255, 3).tolist())

                color = color_map[tid]

                # Use thicker boxes for persistent trackers
                thickness = 3 if tid in persistent_ids else 2

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Add ID label
                label = f"ID {tid}"
                if tid in persistent_ids:
                    label += " ✓"

                frame = put_text_pil(
                    frame, label, (x1, y1 - 10),
                    font_size=20, color=Colors.WHITE,
                    bg_color=color, padding=4
                )

        # Add frame info
        info_text = f"Frame {frame_idx}/{len(frames)} | Detected: {len(detections)}"
        frame = put_text_pil(
            frame, info_text, (10, 10),
            font_size=24, color=Colors.WHITE,
            bg_color=Colors.OVERLAY_DARK, padding=8
        )

        out.write(frame)

    out.release()
    print(f"Visualization saved: {output_path}")


def test_player_tracking(
    video_path: str,
    output_dir: str,
    max_seconds: float = 30.0,
):
    """Test player detection and tracking only."""
    print("=" * 80)
    print("PLAYER TRACKING TEST")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Max duration: {max_seconds}s")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    print("Loading video...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_frames = int(max_seconds * fps) if max_seconds else None

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break
    cap.release()

    print(f"Loaded {len(frames)} frames ({len(frames)/fps:.1f}s)")
    print()

    # Initialize detector with ByteTrack
    print("Initializing RF-DETR + ByteTrack...")
    detector = PlayerRefereeDetector()
    model_type = "RF-DETR (local)" if detector._use_rfdetr else "Roboflow API"
    print(f"Detection model: {model_type}")
    print()

    # Run detection + tracking on all frames
    print("Running detection and tracking...")
    frame_detections = {}
    detections_per_frame = []

    for frame_idx in tqdm(range(len(frames))):
        frame = frames[frame_idx]
        detections = detector.detect_and_track(frame)

        # Filter to only players (not referees)
        if len(detections) > 0:
            player_mask = np.isin(detections.class_id, PLAYER_CLASS_IDS)
            detections = detections[player_mask]

        frame_detections[frame_idx] = detections
        detections_per_frame.append(len(detections))

    print()

    # Analyze tracking quality
    print("Analyzing tracking quality...")
    quality = analyze_tracking_quality(frame_detections)

    print("=" * 80)
    print("TRACKING QUALITY METRICS")
    print("=" * 80)
    print(f"Total unique tracker IDs: {quality['total_trackers']}")
    print(f"Persistent trackers (>30 frames, >80% lifetime): {quality['persistent_trackers']}")
    print(f"Short-lived trackers (<10 frames): {quality['short_lived_trackers']}")
    print(f"Average appearances per tracker: {quality['avg_appearances']:.1f} frames")
    print()

    # Show persistent tracker details
    if quality['persistent_tracker_ids']:
        print("Persistent Tracker IDs (likely real players):")
        for tid in sorted(quality['persistent_tracker_ids']):
            appearances = quality['tracker_appearances'][tid]
            print(f"  ID {tid}: {appearances} frames ({appearances/len(frames)*100:.1f}% of video)")
    print()

    # Detection consistency
    avg_detections = np.mean(detections_per_frame)
    std_detections = np.std(detections_per_frame)
    min_detections = np.min(detections_per_frame)
    max_detections = np.max(detections_per_frame)

    print("DETECTION CONSISTENCY")
    print("=" * 80)
    print(f"Average players per frame: {avg_detections:.1f} ± {std_detections:.1f}")
    print(f"Min/Max players per frame: {min_detections} / {max_detections}")
    print()

    # Check for frames with no detections
    empty_frames = sum(1 for count in detections_per_frame if count == 0)
    if empty_frames > 0:
        print(f"WARNING: {empty_frames} frames with no detections ({empty_frames/len(frames)*100:.1f}%)")
    else:
        print("✓ All frames have at least one detection")
    print()

    # Create visualization
    output_video = os.path.join(output_dir, "tracking_test.mp4")
    create_tracking_visualization(frames, frame_detections, output_video, quality)
    print()

    # Save detailed metrics
    metrics_path = os.path.join(output_dir, "tracking_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("TRACKING QUALITY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames analyzed: {len(frames)} ({len(frames)/fps:.1f}s)\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total unique tracker IDs: {quality['total_trackers']}\n")
        f.write(f"Persistent trackers: {quality['persistent_trackers']}\n")
        f.write(f"Short-lived trackers: {quality['short_lived_trackers']}\n")
        f.write(f"Average detections per frame: {avg_detections:.1f} ± {std_detections:.1f}\n\n")

        f.write("ALL TRACKER IDs (sorted by appearances)\n")
        f.write("-" * 80 + "\n")
        for tid, count in sorted(quality['tracker_appearances'].items(),
                                 key=lambda x: x[1], reverse=True):
            persistent = "✓ PERSISTENT" if tid in quality['persistent_tracker_ids'] else ""
            f.write(f"ID {tid:3d}: {count:4d} frames ({count/len(frames)*100:5.1f}%) {persistent}\n")

    print(f"Detailed metrics saved: {metrics_path}")

    # Final assessment
    print()
    print("=" * 80)
    print("ASSESSMENT (Focus: Detection Consistency)")
    print("=" * 80)
    print()
    print("Note: ID switching is acceptable as long as players are detected consistently.")
    print()

    issues = []
    warnings = []

    # Critical: Are players being detected consistently?
    if empty_frames > len(frames) * 0.05:  # More than 5% empty
        issues.append(f"{empty_frames} frames with no detections ({empty_frames/len(frames)*100:.1f}%)")

    if min_detections == 0:
        issues.append("Some frames have zero detections - players are disappearing")

    if std_detections > avg_detections * 0.6:  # Very high variance
        issues.append(f"High detection variance ({std_detections:.1f}) - inconsistent player detection")

    # Informational: ID tracking quality (less critical)
    if quality['short_lived_trackers'] > quality['persistent_trackers'] * 3:
        warnings.append(f"Many short-lived IDs ({quality['short_lived_trackers']}) - ID switching is high but OK")

    if quality['persistent_trackers'] < 8:
        warnings.append(f"Only {quality['persistent_trackers']} persistent IDs (but detection may still be good)")

    # Display results
    if issues:
        print("❌ DETECTION ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("Action: Players are not being detected consistently. Check RF-DETR model quality.")
    else:
        print("✅ DETECTION QUALITY LOOKS GOOD")
        print(f"  - Players detected in {len(frames) - empty_frames}/{len(frames)} frames ({(len(frames)-empty_frames)/len(frames)*100:.1f}%)")
        print(f"  - Average {avg_detections:.1f} ± {std_detections:.1f} players per frame")
        print(f"  - Min/Max: {min_detections}-{max_detections} detections per frame")

    if warnings:
        print()
        print("ℹ️  INFO (not critical):")
        for warning in warnings:
            print(f"  - {warning}")

    print()
    print("Summary:")
    print(f"  Total unique IDs: {quality['total_trackers']}")
    print(f"  Persistent IDs (>30 frames): {quality['persistent_trackers']}")
    print(f"  Short-lived IDs (<10 frames): {quality['short_lived_trackers']}")
    print()
    print("Review the output video to visually verify that all visible players are detected.")


def main():
    # Use relative paths
    project_root = backend_path.parent
    video_path = str(project_root / "test_videos" / "test_game.mp4")
    output_dir = str(backend_path / "tests" / "outputs" / "tracking_test")

    test_player_tracking(
        video_path=video_path,
        output_dir=output_dir,
        max_seconds=30.0,  # Test first 30 seconds
    )


if __name__ == "__main__":
    main()
