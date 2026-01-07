from app.ml.player_tracker import PlayerTracker
from app.ml.court_detector import CourtDetector

print("Loading models...")
court_detector = CourtDetector()
tracker = PlayerTracker()
tracker.set_court_detector(court_detector)
print("Models loaded successfully!")

video_path = "../test_videos/test_game.mp4"
output_dir = "../outputs/team_tracking"

print(f"\nProcessing video: {video_path}")
print("Using SAM2 Full Video Tracking with Team Classification...")
print("This will take a while...\n")

results = tracker.process_video_with_tracking(
    video_path=video_path,
    output_dir=output_dir,
    num_sample_frames=3
)

print(f"\nResults:")
print(f"  Video: {results.get('video_path')}")
print(f"  Frames: {results.get('frames_processed')}")
print(f"  Players tracked: {results.get('players_tracked')}")
print(f"  Sample frames: {results.get('saved_frames')}")

print(f"\nTeam assignments:")
for obj_id, info in results.get('tracking_info', {}).items():
    team_name = info.get('team_name', info.get('class', 'unknown'))
    print(f"  #{obj_id}: {team_name}")