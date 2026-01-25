"""
Run full 6-stage SwishVision pipeline with ByteTrack + SAM2.

Pipeline: RF-DETR detection → ByteTrack tracking → SAM2 segmentation

Start fresh (resume=False) with improved tracking:
- 30 object capacity (vs 15) to avoid dropping players
- Court mask filtering to exclude spectators
"""
from app.ml.player_tracker import PlayerTracker

# Initialize tracker
tracker = PlayerTracker()

# Run full 6-stage pipeline
result = tracker.process_video_with_tracking(
    video_path='../test_videos/test_game.mp4',
    output_dir='portfolio_outputs_full',
    team_names=('Indiana Pacers', 'Oklahoma City Thunder'),
    use_bytetrack=True,              # ByteTrack for persistent IDs
    use_sam2_segmentation=True,      # SAM2 for pixel-perfect masks
    use_streaming_sam2=True,         # Streaming mode (falls back to batch if unavailable)
    max_total_objects=30,            # Track up to 30 objects (increased from 15)
    use_court_mask_filter=True,      # Filter to only on-court detections
    resume=False                     # Start fresh (set to True to resume from checkpoint)
)

print('\n=== Pipeline Complete ===')
print('Stage videos:', result.get('stage_videos', {}))

# If teams are swapped, you can re-run stages 3-6 with:
# tracker.team_classifier.swap_teams()
# result = tracker.process_video_with_tracking(..., stages_to_generate=[3,4,5,6], resume=True)
