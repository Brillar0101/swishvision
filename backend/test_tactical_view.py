from app.ml.player_tracker import PlayerTracker

print("Loading models...")
tracker = PlayerTracker()
print("Models loaded!")

print("\nProcessing 8 seconds with SAM2 tracking...")
results = tracker.process_video_with_tracking(
    video_path="../test_videos/test_game.mp4",
    output_dir="../outputs/tactical_view",
    max_seconds=8.0,
    num_sample_frames=3
)

print(f"\nDone!")
print(f"Video: {results.get('video_path')}")
print(f"Players: {results.get('players_tracked')}")
