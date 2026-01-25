"""
Run full SwishVision pipeline locally with ByteTrack + SAM2.

Pipeline: RF-DETR detection → ByteTrack tracking → SAM2 segmentation
"""
from app.ml.player_tracker import PlayerTracker

# Initialize tracker
tracker = PlayerTracker()

# Run full pipeline with ByteTrack + SAM2
result = tracker.process_video_with_tracking(
    video_path='../test_videos/test_game.mp4',
    output_dir='portfolio_outputs_streaming',
    team_names=('Indiana Pacers', 'Oklahoma City Thunder'),
    use_bytetrack=True,              # ByteTrack for persistent IDs
    use_sam2_segmentation=True,      # SAM2 for pixel-perfect masks
    use_streaming_sam2=True,         # Streaming mode (falls back to batch if unavailable)
    stages_to_generate=[1, 2],       # Start with first 2 stages
    resume=False
)

print('\n=== Pipeline Complete ===')
print('Stage videos:', result.get('stage_videos', {}))

# After pipeline runs, you can swap teams if needed:
# tracker.team_classifier.swap_teams()
# Then resume with: stages_to_generate=[3,4,5,6], resume=True
