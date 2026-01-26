#!/usr/bin/env python3
"""
Test script for jersey number detection with player tracking.
Uses the Roboflow notebook approach with RF-DETR + SmolVLM2.
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(backend_path / '.env')

from app.ml.player_tracker import PlayerTracker

def main():
    # Use relative paths that work on any machine
    project_root = backend_path.parent
    video_path = project_root / "test_videos" / "test_game.mp4"
    output_dir = backend_path / "training" / "outputs" / "jersey_test"

    print("=" * 60)
    print("SwishVision - Jersey Detection Test")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print()

    # Check for Roboflow API key
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key:
        print(f"Roboflow API Key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    else:
        print("WARNING: ROBOFLOW_API_KEY not set - jersey detection may fail")
    print()

    # Initialize tracker with jersey detection enabled
    # Note: Using CPU device to avoid MPS compatibility issues with SAM2
    tracker = PlayerTracker(
        sam2_checkpoint=str(backend_path / "checkpoints" / "sam2.1_hiera_large.pt"),
        sam2_config="sam2.1_hiera_l",
        enable_jersey_detection=True,
        device="cpu",  # Force CPU to avoid MPS/SAM2 datatype issues on Apple Silicon
    )

    # Process video
    result = tracker.process_video_with_tracking(
        video_path=str(video_path),
        output_dir=str(output_dir),
        max_seconds=15.0,  # Process first 15 seconds
        team_names=("Indiana Pacers", "Oklahoma City Thunder"),
        jersey_ocr_interval=5,  # Run OCR every 5 frames
        smooth_tactical=True,
    )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Total frames: {result['total_frames']}")
    print(f"Players tracked: {result['players_tracked']}")
    print(f"Video saved: {result['video_path']}")
    print()

    # Print jersey numbers found
    jersey_numbers = result.get('jersey_numbers', {})
    if jersey_numbers:
        print("Detected Jersey Numbers:")
        for tracker_id, number in jersey_numbers.items():
            info = result['tracking_info'].get(tracker_id, {})
            team = info.get('team_name', 'Unknown')
            player = info.get('player_name', '')
            if player:
                print(f"  Player {tracker_id}: #{number} {player} ({team})")
            else:
                print(f"  Player {tracker_id}: #{number} ({team})")
    else:
        print("No jersey numbers detected (OCR models may not have loaded)")

    print()
    print(f"Output video: {result['video_path']}")
    print("Done!")

if __name__ == "__main__":
    main()
