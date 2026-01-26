#!/usr/bin/env python3
"""
Debug script to test basic tracking with 10 seconds of video.
This will help identify where the tracking pipeline is failing.
"""
from app.ml.player_tracker import PlayerTracker

print("=" * 60)
print("SwishVision - Debug Pipeline Test")
print("=" * 60)
print("Testing with 10 seconds of video...")
print()

# Initialize tracker with CUDA device (cluster has GPUs)
tracker = PlayerTracker(device="cuda")

# Run pipeline WITH SAM2 camera predictor
result = tracker.process_video_with_tracking(
    video_path='../test_videos/test_game.mp4',
    output_dir='debug_output',
    team_names=('Oklahoma City Thunder', 'Indiana Pacers'),  # white=thunder, yellow=pacers
    use_bytetrack=True,
    use_sam2_segmentation=True,
    use_streaming_sam2=True,  # Use camera predictor (now available)
    max_total_objects=20,
    use_court_mask_filter=False,
    max_seconds=10.0,
    resume=True
)

print()
print("=" * 60)
print("Debug Results")
print("=" * 60)

if 'error' in result:
    print(f"ERROR: {result['error']}")
else:
    print(f"Players tracked: {result['players_tracked']}")
    print(f"Total frames: {result['total_frames']}")
    print(f"Tracking info keys: {list(result['tracking_info'].keys())}")
    print(f"Stage videos generated: {list(result.get('stage_videos', {}).keys())}")
    print()

    # Check if any tracking data exists
    if result['players_tracked'] > 0:
        print("✓ Tracking appears to be working!")
        print(f"  Found {result['players_tracked']} players")
    else:
        print("✗ No players were tracked - this is the bug")

    print()
    print(f"Output directory: debug_output/")
    print("Check the stage videos to see what was generated.")
