"""
Tactical View Module for SwishVision
Uses Roboflow sports library for accurate court mapping via homography.
Recalculates homography per frame for accuracy.
"""
import cv2
import numpy as np
import supervision as sv
from typing import Dict, Tuple, Optional, List
from inference import get_model

from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court
from sports import ViewTransformer, MeasurementUnit


class TacticalView:
    """
    Converts player positions from video frame to tactical 2D court view
    using court keypoint detection and homography transformation.
    """
    
    def __init__(
        self,
        keypoint_model_id: str = "basketball-court-detection-2/14",
        keypoint_confidence: float = 0.3,
        anchor_confidence: float = 0.5,
    ):
        self.keypoint_model = get_model(model_id=keypoint_model_id)
        self.keypoint_confidence = keypoint_confidence
        self.anchor_confidence = anchor_confidence
        
        # Court configuration (NBA, in feet)
        self.config = CourtConfiguration(
            league=League.NBA,
            measurement_unit=MeasurementUnit.FEET
        )
        
        # Cache for homography transformer
        self._last_transformer: Optional[ViewTransformer] = None
        self._last_keypoints_xy = None
        self._last_keypoints_conf = None
    
    def detect_keypoints(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Detect court keypoints.
        
        Returns:
            (keypoints_xy, keypoints_confidence, has_enough_points)
        """
        result = self.keypoint_model.infer(frame, confidence=self.keypoint_confidence)[0]
        key_points = sv.KeyPoints.from_inference(result)
        
        xy = key_points.xy[0]
        conf = key_points.confidence[0]
        
        self._last_keypoints_xy = xy
        self._last_keypoints_conf = conf
        
        mask = conf > self.anchor_confidence
        has_enough = np.sum(mask) >= 4
        
        return xy, conf, has_enough
    
    def build_transformer(self, frame: np.ndarray) -> Optional[ViewTransformer]:
        """
        Build homography transformer for a frame.
        Detects keypoints and creates ViewTransformer.
        
        Returns:
            ViewTransformer or None if not enough keypoints
        """
        result = self.keypoint_model.infer(frame, confidence=self.keypoint_confidence)[0]
        key_points = sv.KeyPoints.from_inference(result)
        
        self._last_keypoints_xy = key_points.xy[0]
        self._last_keypoints_conf = key_points.confidence[0]
        
        mask = self._last_keypoints_conf > self.anchor_confidence
        
        if np.sum(mask) >= 4:
            court_landmarks = np.array(self.config.vertices)[mask]
            frame_landmarks = key_points[:, mask].xy[0]
            
            self._last_transformer = ViewTransformer(
                source=frame_landmarks,
                target=court_landmarks
            )
            return self._last_transformer
        
        return self._last_transformer  # Return cached if current frame fails
    
    def transform_to_court(
        self,
        frame: np.ndarray,
        pixel_positions: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Transform pixel positions to court coordinates.
        Builds homography for this frame.
        
        Args:
            frame: Video frame (for keypoint detection)
            pixel_positions: Nx2 array of pixel coordinates
            
        Returns:
            Nx2 array of court coordinates (feet) or None
        """
        if len(pixel_positions) == 0:
            return None
        
        transformer = self.build_transformer(frame)
        if transformer is None:
            return None
        
        return transformer.transform_points(points=pixel_positions.astype(np.float32))
    
    def render(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        teams: np.ndarray,
        team_colors: Dict[int, str] = None,
    ) -> np.ndarray:
        """
        Render tactical court view with player positions.
        
        Args:
            frame: Video frame (for keypoint detection)
            detections: Player detections
            teams: Array of team IDs (0 or 1) matching detections
            team_colors: Dict mapping team_id -> hex color string
            
        Returns:
            Blue tactical court image with players drawn
        """
        if team_colors is None:
            team_colors = {
                0: "#00FF00",  # Green for Team A
                1: "#FF0000",  # Red for Team B
            }
        
        # Create blue court
        court = draw_court(config=self.config)
        court_hsv = cv2.cvtColor(court, cv2.COLOR_BGR2HSV)
        court_hsv[:,:,0] = 110  # Blue hue
        court_hsv[:,:,1] = np.clip(court_hsv[:,:,1] * 1.5, 0, 255).astype(np.uint8)
        court_blue = cv2.cvtColor(court_hsv, cv2.COLOR_HSV2BGR)
        
        if len(detections) == 0:
            return court_blue
        
        # Get player feet positions (bottom center of bbox)
        frame_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        # Transform to court coordinates
        court_xy = self.transform_to_court(frame, frame_xy)
        if court_xy is None:
            return court_blue
        
        # Draw players by team
        for team_id in [0, 1]:
            mask = teams == team_id
            if not np.any(mask):
                continue
            
            color_hex = team_colors.get(team_id, "#808080")
            court_blue = draw_points_on_court(
                config=self.config,
                xy=court_xy[mask],
                fill_color=sv.Color.from_hex(color_hex),
                court=court_blue
            )
        
        return court_blue
    
    def render_simple(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        frame_shape: Tuple[int, int],
        team_assignments: Dict[int, int] = None,
        team_colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Render tactical court view with player positions (dict-based API).
        Uses cached transformer - call build_transformer first!
        
        Args:
            player_positions: Dict mapping obj_id -> (x, y) pixel coordinates
            frame_shape: (height, width) of the video frame
            team_assignments: Dict mapping obj_id -> team_id (0, 1, or -1 for ref)
            team_colors: Dict mapping team_id -> BGR color tuple
            
        Returns:
            Blue tactical court image with players drawn
        """
        if team_colors is None:
            team_colors = {
                0: (0, 255, 0),    # Green
                1: (0, 0, 255),    # Red
                -1: (0, 255, 255), # Yellow (referees)
            }
        
        # Create blue court
        court = draw_court(config=self.config)
        court_hsv = cv2.cvtColor(court, cv2.COLOR_BGR2HSV)
        court_hsv[:,:,0] = 110
        court_hsv[:,:,1] = np.clip(court_hsv[:,:,1] * 1.5, 0, 255).astype(np.uint8)
        court_blue = cv2.cvtColor(court_hsv, cv2.COLOR_HSV2BGR)
        
        if self._last_transformer is None or len(player_positions) == 0:
            return court_blue
        
        # Convert positions dict to array
        obj_ids = list(player_positions.keys())
        positions = np.array([player_positions[oid] for oid in obj_ids], dtype=np.float32)
        
        # Transform to court coordinates
        court_xy = self._last_transformer.transform_points(points=positions)
        if court_xy is None:
            return court_blue
        
        # Draw players by team
        for team_id in [0, 1, -1]:
            team_mask = []
            for i, oid in enumerate(obj_ids):
                if team_assignments is None:
                    t = 0
                else:
                    t = team_assignments.get(oid, 0)
                team_mask.append(t == team_id)
            
            team_mask = np.array(team_mask)
            if not np.any(team_mask):
                continue
            
            color = team_colors.get(team_id, (128, 128, 128))
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
            
            court_blue = draw_points_on_court(
                config=self.config,
                xy=court_xy[team_mask],
                fill_color=sv.Color.from_hex(hex_color),
                court=court_blue
            )
        
        return court_blue
    
    def render_with_numbers(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        frame_shape: Tuple[int, int],
        team_assignments: Dict[int, int] = None,
        team_colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Render tactical court view with numbered circles for players.

        Args:
            player_positions: Dict mapping obj_id -> (x, y) pixel coordinates
            frame_shape: (height, width) of the video frame
            team_assignments: Dict mapping obj_id -> team_id (0, 1, or -1 for ref)
            team_colors: Dict mapping team_id -> BGR color tuple

        Returns:
            Blue tactical court image with numbered player circles
        """
        if team_colors is None:
            team_colors = {
                0: (0, 255, 0),    # Green
                1: (0, 0, 255),    # Red
                -1: (0, 255, 255), # Yellow (referees)
            }

        # Create blue court
        court_blue = self.get_court_image()

        if self._last_transformer is None or len(player_positions) == 0:
            return court_blue

        # Convert positions dict to array
        obj_ids = list(player_positions.keys())
        positions = np.array([player_positions[oid] for oid in obj_ids], dtype=np.float32)

        # Transform to court coordinates
        court_xy = self._last_transformer.transform_points(points=positions)
        if court_xy is None:
            return court_blue

        # Court dimensions for scaling
        court_h, court_w = court_blue.shape[:2]

        # NBA court is 94 x 50 feet
        scale_x = court_w / 94.0
        scale_y = court_h / 50.0

        # Draw each player as a numbered circle
        for i, obj_id in enumerate(obj_ids):
            # Get court position (in feet, centered at 0,0)
            cx_feet, cy_feet = court_xy[i]

            # Convert to pixel coordinates (court image has origin at top-left)
            # Court center is at (47, 25) feet
            px = int((cx_feet + 47) * scale_x)
            py = int((25 - cy_feet) * scale_y)  # Flip Y

            # Skip if outside court bounds
            if px < 0 or px >= court_w or py < 0 or py >= court_h:
                continue

            # Get team color
            team_id = team_assignments.get(obj_id, 0) if team_assignments else 0
            color = team_colors.get(team_id, (128, 128, 128))

            # Draw filled circle
            radius = 18
            cv2.circle(court_blue, (px, py), radius, color, -1)
            cv2.circle(court_blue, (px, py), radius, (255, 255, 255), 2)  # White border

            # Draw player ID inside circle
            number = str(obj_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 if len(number) <= 2 else 0.4
            thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(number, font, font_scale, thickness)
            text_x = px - text_w // 2
            text_y = py + text_h // 2

            cv2.putText(court_blue, number, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return court_blue
    
    def get_court_image(self) -> np.ndarray:
        """Get empty blue court image."""
        court = draw_court(config=self.config)
        court_hsv = cv2.cvtColor(court, cv2.COLOR_BGR2HSV)
        court_hsv[:,:,0] = 110
        court_hsv[:,:,1] = np.clip(court_hsv[:,:,1] * 1.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(court_hsv, cv2.COLOR_HSV2BGR)
    
    def draw_keypoints_on_frame(
        self,
        frame: np.ndarray,
        high_conf_only: bool = True
    ) -> np.ndarray:
        """Draw detected keypoints on frame."""
        annotated = frame.copy()
        
        if self._last_keypoints_xy is None:
            return annotated
        
        for pt, conf in zip(self._last_keypoints_xy, self._last_keypoints_conf):
            if conf < self.keypoint_confidence:
                continue
            if high_conf_only and conf < self.anchor_confidence:
                continue
            
            color = (0, 255, 0) if conf > self.anchor_confidence else (0, 165, 255)
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 5, color, -1)
        
        return annotated


def create_combined_view(
    frame: np.ndarray,
    tactical_view: np.ndarray,
    position: str = 'bottom-right',
    scale_factor: float = 0.35,
) -> np.ndarray:
    """
    Overlay tactical view on video frame.
    
    Args:
        frame: Original video frame
        tactical_view: Tactical court view image
        position: 'bottom-right', 'bottom-left', 'top-right', 'top-left'
        scale_factor: Size of tactical view relative to frame width
        
    Returns:
        Combined frame with tactical overlay
    """
    fh, fw = frame.shape[:2]
    th, tw = tactical_view.shape[:2]
    
    target_width = int(fw * scale_factor)
    scale = target_width / tw
    ntw = int(tw * scale)
    nth = int(th * scale)
    
    tactical_resized = cv2.resize(tactical_view, (ntw, nth))
    
    combined = frame.copy()
    margin = 15
    
    if 'right' in position:
        x = fw - ntw - margin
    else:
        x = margin
    
    if 'top' in position:
        y = margin
    else:
        y = fh - nth - margin
    
    # Semi-transparent background
    overlay = combined.copy()
    cv2.rectangle(overlay, (x - 10, y - 10), (x + ntw + 10, y + nth + 10), (20, 20, 20), -1)
    combined = cv2.addWeighted(overlay, 0.9, combined, 0.1, 0)
    
    # Border
    cv2.rectangle(combined, (x - 10, y - 10), (x + ntw + 10, y + nth + 10), (80, 80, 80), 2)
    
    # Place tactical view
    combined[y:y + nth, x:x + ntw] = tactical_resized
    
    # Label
    cv2.putText(combined, "TACTICAL VIEW", (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return combined