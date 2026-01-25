"""
Run SwishVision pipeline WITHOUT SAM2 segmentation.

Pipeline: RF-DETR detection → ByteTrack tracking (no pixel-perfect masks)

This uses much less memory and should complete successfully.
SAM2 can be added later once we confirm tracking works.
"""
from app.ml.player_tracker import PlayerTracker

# Initialize tracker
tracker = PlayerTracker()

# Run pipeline WITHOUT SAM2 (much lower memory usage)
result = tracker.process_video_with_tracking(
    video_path='../test_videos/test_game.mp4',
    output_dir='portfolio_outputs_no_sam2',
    team_names=('Indiana Pacers', 'Oklahoma City Thunder'),
    use_bytetrack=True,              # ByteTrack for persistent IDs
    use_sam2_segmentation=False,     # NO SAM2 - uses bounding box masks instead
    max_total_objects=30,            # Can use higher capacity without SAM2
    use_court_mask_filter=True,      # Filter to only on-court detections
    max_seconds=15.0,                # Process first 15 seconds
    resume=False                     # Start fresh
)

print('\n=== Pipeline Complete ===')
print('Stage videos:', result.get('stage_videos', {}))
print('\nNote: This run uses bounding box masks instead of SAM2 pixel-perfect masks.')
print('Once this completes, we can test SAM2 separately if needed.')
