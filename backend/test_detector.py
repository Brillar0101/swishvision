from app.ml.player_detector import PlayerDetector

detector = PlayerDetector()
print("Model loaded successfully!")

results = detector.process_video("../test_videos/test_game.mp4", "../outputs/playerdetection", num_frames=3)

print(f"Processed {results['frames_processed']} frames")
for r in results["results"]:
    print(f"Frame {r['frame_number']}: {len(r['detections'])} players detected")