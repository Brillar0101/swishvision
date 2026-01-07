from app.ml.player_tracker import PlayerTracker
import cv2
import os

print("Loading models...")
tracker = PlayerTracker()

# Just process and save 3 sample frames
results = tracker.process_video_with_tracking(
    video_path="../test_videos/test_game.mp4",
    output_dir="../outputs/tactical_view",
    max_seconds=3.0,
    num_sample_frames=3
)

print(f"\nDone! Frames saved to: ../outputs/tactical_view/")
