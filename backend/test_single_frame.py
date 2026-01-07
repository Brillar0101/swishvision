"""
Test tactical view with a single frame
"""
import cv2
import sys
sys.path.insert(0, '/Users/barakaeli/Desktop/Github Projects/swishvision/backend')

from app.ml.tactical_view import TacticalView, create_combined_view

# Load test frame - correct path
frame = cv2.imread('/Users/barakaeli/Desktop/Github Projects/swishvision/outputs/playerdetection_tracked/playertracking_frame_02.jpg')

if frame is None:
    print("Could not load frame!")
    exit(1)

print(f"Frame size: {frame.shape}")

# Create tactical view (half-court mode)
tv = TacticalView(view_mode='half')
print(f"Tactical view size: {tv.width}x{tv.height}")

# Sample player positions (from the frame - approximate positions)
# Format: player_id -> (x, y) in frame coordinates
player_positions = {
    2: (190, 350),    # Referee left
    14: (350, 290),   # Red player
    8: (480, 280),    # Orange player
    11: (420, 350),   # Red player
    6: (540, 360),    # Orange player
    3: (360, 400),    # Red player
    12: (850, 250),   # Red player
    4: (870, 340),    # Orange player
    1: (1070, 280),   # Referee right
    13: (1020, 470),  # Orange player
    9: (920, 560),    # Referee bottom
}

# Team assignments: 0 = Orange (Indiana), 1 = Red (OKC), -1 = Referee
team_assignments = {
    2: -1,   # Referee
    14: 1,   # Red
    8: 0,    # Orange
    11: 1,   # Red
    6: 0,    # Orange
    3: 1,    # Red
    12: 1,   # Red
    4: 0,    # Orange
    1: -1,   # Referee
    13: 0,   # Orange
    9: -1,   # Referee
}

# Render tactical view
tactical = tv.render(
    player_positions=player_positions,
    frame=frame,
    team_assignments=team_assignments
)

# Save outputs
output_dir = '/Users/barakaeli/Desktop/Github Projects/swishvision/outputs'

cv2.imwrite(f'{output_dir}/tactical_only.jpg', tactical)
print(f"Saved: {output_dir}/tactical_only.jpg")

combined = create_combined_view(frame, tactical, position='bottom-right')
cv2.imwrite(f'{output_dir}/combined_view.jpg', combined)
print(f"Saved: {output_dir}/combined_view.jpg")

cv2.imwrite(f'{output_dir}/court_template.jpg', tv.court_image)
print(f"Saved: {output_dir}/court_template.jpg")

print("\nDone!")