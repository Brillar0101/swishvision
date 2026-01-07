from app.ml.player_tracker import PlayerTracker

print("="*60)
print("Full Video Tracking with Jersey Numbers - FULL VIDEO")
print("="*60)

tracker = PlayerTracker(enable_jersey_numbers=True)

results = tracker.process_video_with_tracking(
    video_path="../test_videos/test_game.mp4",
    output_dir="../outputs/full_tracking",
    max_seconds=9999.0,  # No time limit - process entire video
    detect_jersey_numbers=True
)

print("\nDone!")
print(f"Video: {results['video_path']}")
print(f"Total frames: {results['total_frames']}")
print(f"Players tracked: {results['players_tracked']}")
print(f"Jersey numbers: {results['jersey_numbers']}")
