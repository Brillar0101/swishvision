"""
Run full SwishVision pipeline locally with team swap.
"""
from app.ml.player_tracker import PlayerTracker

# Initialize tracker
tracker = PlayerTracker()

# Swap teams to fix incorrect assignment
tracker.team_classifier.swap_teams()

# Run full pipeline
result = tracker.process_video_with_tracking(
    video_path='../test_videos/test_game.mp4',
    output_dir='portfolio_outputs_local',
    team_names=('Indiana Pacers', 'Oklahoma City Thunder'),
    resume=False
)

print('\n=== Pipeline Complete ===')
print('Stage videos:', result.get('stage_videos', {}))
