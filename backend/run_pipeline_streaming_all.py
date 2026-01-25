"""
Run full 6-stage SwishVision pipeline with streaming SAM2.

Pipeline: RF-DETR detection → ByteTrack tracking → SAM2 streaming segmentation
"""
from app.ml.player_tracker import PlayerTracker

# Initialize tracker
tracker = PlayerTracker()

# Swap teams to fix incorrect assignment
tracker.team_classifier.swap_teams()

# Run full 6-stage pipeline with streaming SAM2
result = tracker.process_video_with_tracking(
    video_path='../test_videos/test_game.mp4',
    output_dir='portfolio_outputs_streaming',
    team_names=('Indiana Pacers', 'Oklahoma City Thunder'),
    use_bytetrack=True,              # ByteTrack for persistent IDs
    use_sam2_segmentation=True,      # SAM2 for pixel-perfect masks
    use_streaming_sam2=True,         # Streaming mode (lower memory)
    resume=True                      # Resume from existing progress
)

print('\n=== Pipeline Complete ===')
print('Stage videos:', result.get('stage_videos', {}))
