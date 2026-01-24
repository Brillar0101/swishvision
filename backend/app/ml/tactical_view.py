"""
Tactical View Module - 2D court view with player positions.

Uses the Roboflow sports library for court drawing and homography transformation.
Based on the basketball AI notebook from Roboflow.

Install the sports library with:
    pip install git+https://github.com/roboflow/sports.git@feat/basketball
"""
import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

import supervision as sv

try:
    from app.ml.ui_config import Colors, TextSize, put_text_pil
    UI_CONFIG_AVAILABLE = True
except ImportError:
    # Fallback for standalone execution
    UI_CONFIG_AVAILABLE = False
    Colors = None

# Try to import from roboflow sports library (basketball branch)
try:
    from sports import ViewTransformer, MeasurementUnit
    from sports.basketball import CourtConfiguration, League
    from sports.basketball import draw_court as _sports_draw_court
    from sports.basketball import draw_points_on_court as _draw_points_on_court
    SPORTS_LIBRARY_AVAILABLE = True
except ImportError:
    SPORTS_LIBRARY_AVAILABLE = False
    ViewTransformer = None
    print("Warning: sports library not available. Install with:")
    print("  pip install git+https://github.com/roboflow/sports.git@feat/basketball")

# Court keypoint model
COURT_KEYPOINT_MODEL_ID = "basketball-court-detection-2/19"
KEYPOINT_CONFIDENCE = 0.3
ANCHOR_CONFIDENCE = 0.5

# Player class IDs (from player_referee_detector)
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]  # player variants
REFEREE_CLASS_IDS = [8]  # referee

# Team colors (BGR format for OpenCV)
TEAM_COLORS = {
    0: (0, 255, 0),    # GREEN for Team 1
    1: (0, 0, 255),    # RED for Team 2
    -1: (0, 255, 255), # YELLOW for referees
}


def get_positions_from_detections(
    detections: sv.Detections,
    anchor: sv.Position = sv.Position.BOTTOM_CENTER
) -> Dict[int, Tuple[float, float]]:
    """
    Extract player positions from sv.Detections.

    Uses the bottom center of bounding boxes (foot position) by default.

    Args:
        detections: Supervision Detections object with tracker_id
        anchor: Position anchor to use (default: BOTTOM_CENTER for foot position)

    Returns:
        Dict mapping tracker_id to (x, y) frame pixel positions
    """
    if len(detections) == 0 or detections.tracker_id is None:
        return {}

    # Get anchor coordinates (bottom center = foot position)
    xy = detections.get_anchors_coordinates(anchor=anchor)

    positions = {}
    for i, tracker_id in enumerate(detections.tracker_id):
        positions[int(tracker_id)] = (float(xy[i, 0]), float(xy[i, 1]))

    return positions


def get_team_from_class_id(class_id: int) -> int:
    """
    Determine if a detection is a player or referee based on class_id.

    Args:
        class_id: Detection class ID

    Returns:
        -1 for referee, 0 for player (team assignment done separately)
    """
    if class_id in REFEREE_CLASS_IDS:
        return -1  # Referee
    return 0  # Player (default team, actual team assigned by TeamClassifier)


class TacticalView:
    """
    Converts player positions from video frame to tactical 2D court view
    using homography transformation.

    Based on Roboflow basketball AI notebook implementation.
    """

    def __init__(
        self,
        keypoint_model_id: str = COURT_KEYPOINT_MODEL_ID,
        keypoint_confidence: float = KEYPOINT_CONFIDENCE,
        anchor_confidence: float = ANCHOR_CONFIDENCE,
    ):
        """
        Initialize TacticalView.

        Args:
            keypoint_model_id: Roboflow model ID for court keypoint detection
            keypoint_confidence: Minimum confidence for keypoint detection
            anchor_confidence: Minimum confidence for anchor points used in homography
        """
        self.keypoint_model_id = keypoint_model_id
        self.keypoint_confidence = keypoint_confidence
        self.anchor_confidence = anchor_confidence

        self._keypoint_model = None
        self._last_transformer = None  # ViewTransformer when available

        # Get court configuration from sports library
        if SPORTS_LIBRARY_AVAILABLE:
            self.config = CourtConfiguration(
                league=League.NBA,
                measurement_unit=MeasurementUnit.FEET
            )
        else:
            self.config = None

    def _load_model(self):
        """Lazy load keypoint detection model."""
        if self._keypoint_model is not None:
            return

        try:
            from inference import get_model

            # Load .env with override to ensure we get the correct API key
            load_dotenv(override=True)
            api_key = os.getenv("ROBOFLOW_API_KEY")

            self._keypoint_model = get_model(
                model_id=self.keypoint_model_id,
                api_key=api_key
            )
        except Exception as e:
            print(f"  Failed to load court keypoint model: {e}")

    def build_transformer(self, frame: np.ndarray):
        """
        Build homography transformer for a frame by detecting court keypoints.

        Args:
            frame: Video frame (BGR)

        Returns:
            ViewTransformer if successful, None otherwise
        """
        if not SPORTS_LIBRARY_AVAILABLE:
            print("  Sports library not available for homography")
            return self._last_transformer

        self._load_model()
        if self._keypoint_model is None:
            return self._last_transformer

        try:
            # Detect court keypoints
            result = self._keypoint_model.infer(frame, confidence=self.keypoint_confidence)[0]
            key_points = sv.KeyPoints.from_inference(result)

            # Filter to high-confidence anchor points
            landmarks_mask = key_points.confidence[0] > self.anchor_confidence

            if np.count_nonzero(landmarks_mask) >= 4:
                # Get court landmarks from config (real-world coordinates)
                court_landmarks = np.array(self.config.vertices)[landmarks_mask]
                # Get frame landmarks (pixel coordinates)
                frame_landmarks = key_points[:, landmarks_mask].xy[0]

                # Build homography transformer
                self._last_transformer = ViewTransformer(
                    source=frame_landmarks,
                    target=court_landmarks,
                )
                return self._last_transformer

        except Exception as e:
            print(f"  Error building transformer: {e}")

        return self._last_transformer

    def transform_positions(
        self,
        frame_positions: Dict[int, Tuple[float, float]],
    ) -> Dict[int, Tuple[float, float]]:
        """
        Transform player positions from frame coordinates to court coordinates.

        Args:
            frame_positions: Dict mapping object IDs to (x, y) frame positions

        Returns:
            Dict mapping object IDs to (x, y) court positions in feet
        """
        if self._last_transformer is None or len(frame_positions) == 0:
            return {}

        obj_ids = list(frame_positions.keys())
        positions = np.array([frame_positions[oid] for oid in obj_ids], dtype=np.float32)

        # Transform to court coordinates
        court_xy = self._last_transformer.transform_points(points=positions)

        return {oid: (court_xy[i, 0], court_xy[i, 1]) for i, oid in enumerate(obj_ids)}

    def render_from_detections(
        self,
        detections: sv.Detections,
        team_assignments: Optional[Dict[int, int]] = None,
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Render tactical court view directly from sv.Detections.

        This is a convenience method that extracts positions and renders in one call.

        Args:
            detections: Supervision Detections with tracker_id
            team_assignments: Dict mapping tracker_id to team IDs (0, 1, or -1 for referee)
            team_colors: Dict mapping team IDs to BGR colors

        Returns:
            Court image with player positions drawn (BGR)
        """
        # Extract positions from detections
        positions = get_positions_from_detections(detections)

        # If no team assignments provided, use class_id to identify referees
        if team_assignments is None and len(detections) > 0 and detections.tracker_id is not None:
            team_assignments = {}
            for i, tracker_id in enumerate(detections.tracker_id):
                class_id = detections.class_id[i]
                team_assignments[int(tracker_id)] = get_team_from_class_id(class_id)

        return self.render(positions, (0, 0), team_assignments, team_colors)

    def render(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        frame_shape: Tuple[int, int],  # Kept for API compatibility
        team_assignments: Optional[Dict[int, int]] = None,
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Render tactical court view with player positions.

        Args:
            player_positions: Dict mapping object IDs to (x, y) frame pixel positions
            frame_shape: (height, width) of the frame (kept for API compatibility)
            team_assignments: Dict mapping object IDs to team IDs (0, 1, or -1 for referee)
            team_colors: Dict mapping team IDs to BGR colors

        Returns:
            Court image with player positions drawn (BGR)
        """
        _ = frame_shape  # Unused but kept for backwards compatibility

        if team_colors is None:
            team_colors = TEAM_COLORS

        if not SPORTS_LIBRARY_AVAILABLE:
            # Return empty court placeholder
            court = np.ones((500, 940, 3), dtype=np.uint8) * 60
            court[:, :, 0] = 180  # Blue tint
            cv2.putText(court, "Sports library not installed", (200, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return court

        # Draw base court using sports library
        court_img = _sports_draw_court(config=self.config)

        if self._last_transformer is None or len(player_positions) == 0:
            return court_img

        # Transform positions from frame to court coordinates
        obj_ids = list(player_positions.keys())
        positions = np.array([player_positions[oid] for oid in obj_ids], dtype=np.float32)

        court_xy = self._last_transformer.transform_points(points=positions)

        if team_assignments is None:
            team_assignments = {oid: 0 for oid in obj_ids}

        # Group positions by team for drawing
        for team_id in set(team_assignments.values()):
            # Get positions for this team
            team_mask = np.array([team_assignments.get(oid, 0) == team_id for oid in obj_ids])
            if not np.any(team_mask):
                continue

            team_xy = court_xy[team_mask]

            # Get color for this team (convert BGR to sv.Color)
            bgr_color = team_colors.get(team_id, (128, 128, 128))
            # sv.Color uses RGB format
            fill_color = sv.Color(r=bgr_color[2], g=bgr_color[1], b=bgr_color[0])

            # Draw points on court
            court_img = _draw_points_on_court(
                config=self.config,
                xy=team_xy,
                fill_color=fill_color,
                court=court_img
            )

        return court_img


def draw_court(config=None, width: int = 940, height: int = 500) -> np.ndarray:
    """
    Draw an NBA basketball court.

    Uses the sports library if available, otherwise draws a simple court.

    Args:
        config: CourtConfiguration (optional, uses NBA default if not provided)
        width: Image width in pixels (used for fallback)
        height: Image height in pixels (used for fallback)

    Returns:
        Court image (BGR)
    """
    if SPORTS_LIBRARY_AVAILABLE:
        if config is None:
            config = CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)
        return _sports_draw_court(config=config)

    # Fallback: draw simple court
    court = np.ones((height, width, 3), dtype=np.uint8)
    court[:, :] = (180, 120, 60)  # Blue background

    # Court outline
    cv2.rectangle(court, (10, 10), (width - 10, height - 10), (255, 255, 255), 2)

    # Center line
    cv2.line(court, (width // 2, 10), (width // 2, height - 10), (255, 255, 255), 2)

    # Center circle
    cv2.circle(court, (width // 2, height // 2), 60, (255, 255, 255), 2)

    return court


def create_combined_view(
    frame: np.ndarray,
    tactical_view: np.ndarray,
    position: str = 'bottom-right',
    scale_factor: float = 0.35,
) -> np.ndarray:
    """
    Overlay tactical view on video frame.

    Args:
        frame: Video frame (BGR)
        tactical_view: Tactical court view (BGR)
        position: Position of overlay ('bottom-right', 'bottom-left', 'top-right', 'top-left')
        scale_factor: Size of overlay relative to frame width

    Returns:
        Combined frame with tactical view overlay
    """
    fh, fw = frame.shape[:2]
    th, tw = tactical_view.shape[:2]

    # Calculate target size
    target_width = int(fw * scale_factor)
    scale = target_width / tw
    ntw = int(tw * scale)
    nth = int(th * scale)

    # Resize tactical view
    tactical_resized = cv2.resize(tactical_view, (ntw, nth))

    # Create copy of frame
    combined = frame.copy()
    margin = 15

    # Calculate position
    if 'right' in position:
        x = fw - ntw - margin
    else:
        x = margin

    if 'bottom' in position:
        y = fh - nth - margin
    else:
        y = margin

    # Add professional border
    border_color = Colors.WHITE if UI_CONFIG_AVAILABLE else (255, 255, 255)
    cv2.rectangle(combined, (x - 3, y - 3), (x + ntw + 3, y + nth + 3), border_color, 3)

    # Overlay tactical view
    combined[y:y + nth, x:x + ntw] = tactical_resized

    # Add professional label
    if UI_CONFIG_AVAILABLE:
        combined = put_text_pil(
            combined, "TACTICAL VIEW", (x + 10, y - 40),
            font_size=TextSize.SMALL, color=Colors.WHITE,
            bg_color=Colors.OVERLAY_DARK, padding=8
        )
    else:
        cv2.putText(combined, "TACTICAL VIEW", (x + 5, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return combined
