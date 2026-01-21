"""
Tactical View Module for SwishVision
Uses roboflow sports library for basketball court configuration.
"""
import cv2
import numpy as np
import supervision as sv
from typing import Dict, Tuple, Optional, List

# Try to import from sports library (roboflow)
try:
    from sports.basketball import CourtConfiguration, League, MeasurementUnit, draw_court as sports_draw_court
    SPORTS_AVAILABLE = True
except ImportError:
    SPORTS_AVAILABLE = False
    print("Warning: sports library not available. Install with: pip install git+https://github.com/roboflow/sports.git@feat/basketball")

# NBA Court dimensions in feet
NBA_COURT_LENGTH = 94.0
NBA_COURT_WIDTH = 50.0


class ViewTransformer:
    """Homography-based coordinate transformer."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initialize transformer with point correspondences.

        Args:
            source: Nx2 array of source points (frame coordinates)
            target: Nx2 array of target points (court coordinates)
        """
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.homography, _ = cv2.findHomography(self.source, self.target)

    def transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Transform points from source to target coordinates."""
        if self.homography is None or len(points) == 0:
            return None

        points = points.astype(np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.homography)
        return transformed.reshape(-1, 2)


def draw_court(width: int = 940, height: int = 500) -> np.ndarray:
    """
    Draw an NBA basketball court.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Court image (BGR)
    """
    court = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray

    # Scale factors
    sx = width / NBA_COURT_LENGTH
    sy = height / NBA_COURT_WIDTH

    def to_px(x_feet, y_feet):
        px = int((x_feet + NBA_COURT_LENGTH / 2) * sx)
        py = int((NBA_COURT_WIDTH / 2 - y_feet) * sy)
        return px, py

    # Court outline
    cv2.rectangle(court, to_px(-47, 25), to_px(47, -25), (255, 255, 255), 2)

    # Center circle
    cv2.circle(court, to_px(0, 0), int(6 * sx), (255, 255, 255), 2)

    # Center line
    cv2.line(court, to_px(0, 25), to_px(0, -25), (255, 255, 255), 2)

    # Three point lines
    for side in [-1, 1]:
        # Corner threes
        cv2.line(court, to_px(side * 47, 22), to_px(side * 33, 22), (255, 255, 255), 2)
        cv2.line(court, to_px(side * 47, -22), to_px(side * 33, -22), (255, 255, 255), 2)
        # Arc
        center = to_px(side * 41.75, 0)
        cv2.ellipse(court, center, (int(23.75 * sx), int(23.75 * sy)), 0,
                   90 if side > 0 else -90, 270 if side > 0 else 90, (255, 255, 255), 2)

    # Free throw lanes (paint)
    for side in [-1, 1]:
        cv2.rectangle(court, to_px(side * 47, 8), to_px(side * 28, -8), (255, 255, 255), 2)
        # Free throw circle
        cv2.circle(court, to_px(side * 28, 0), int(6 * sx), (255, 255, 255), 2)

    # Restricted areas
    for side in [-1, 1]:
        center = to_px(side * 41.75, 0)
        cv2.ellipse(court, center, (int(4 * sx), int(4 * sy)), 0, 0, 360, (255, 255, 255), 2)

    # Baskets
    for side in [-1, 1]:
        cv2.circle(court, to_px(side * 41.75, 0), int(0.75 * sx), (255, 128, 0), -1)

    return court


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
        self.keypoint_model_id = keypoint_model_id
        self.keypoint_confidence = keypoint_confidence
        self.anchor_confidence = anchor_confidence

        self._keypoint_model = None
        self._last_transformer: Optional[ViewTransformer] = None
        self._last_keypoints_xy = None
        self._last_keypoints_conf = None

        # Court vertices for homography - use sports library if available
        if SPORTS_AVAILABLE:
            config = CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)
            self.vertices = np.array(config.vertices, dtype=np.float32)
            print(f"  Loaded {len(self.vertices)} court vertices from sports library")
        else:
            # Fallback vertices (may not match model exactly)
            print("  WARNING: Using fallback court vertices - install sports library for accuracy")
            self.vertices = self._get_fallback_vertices()

    def _get_fallback_vertices(self) -> np.ndarray:
        """Fallback court vertices when sports library is not available."""
        # These are approximate NBA court landmark positions in feet
        # Origin at center court, x along length, y along width
        return np.array([
            [0, 0],       # Center court
            [-47, 25],    # Corner
            [-47, -25],
            [47, 25],
            [47, -25],
            [-47, 8],     # Paint corners
            [-47, -8],
            [-28, 8],
            [-28, -8],
            [-28, 0],
            [47, 8],
            [47, -8],
            [28, 8],
            [28, -8],
            [28, 0],
            [-22, 25],    # 3-point line
            [-22, -25],
            [22, 25],
            [22, -25],
            [-23.75, 8],
            [-23.75, -8],
            [23.75, 8],
            [23.75, -8],
            [0, 25],      # Half court
            [0, -25],
            [0, 6],
            [0, -6],
            [-47, 0],     # Baseline center
            [47, 0],
            [-28, 6],     # Free throw
            [-28, -6],
            [28, 6],
            [28, -6],
        ], dtype=np.float32)

    def _load_model(self):
        """Lazy load keypoint detection model."""
        if self._keypoint_model is not None:
            return

        try:
            from inference import get_model
            self._keypoint_model = get_model(model_id=self.keypoint_model_id)
            print("  Court keypoint model loaded")
        except Exception as e:
            print(f"  Failed to load court keypoint model: {e}")

    def detect_keypoints(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """Detect court keypoints."""
        self._load_model()
        if self._keypoint_model is None:
            return None, None, False

        result = self._keypoint_model.infer(frame, confidence=self.keypoint_confidence)[0]
        key_points = sv.KeyPoints.from_inference(result)

        xy = key_points.xy[0]
        conf = key_points.confidence[0]

        self._last_keypoints_xy = xy
        self._last_keypoints_conf = conf

        mask = conf > self.anchor_confidence
        has_enough = np.sum(mask) >= 4

        return xy, conf, has_enough

    def build_transformer(self, frame: np.ndarray) -> Optional[ViewTransformer]:
        """Build homography transformer for a frame."""
        self._load_model()
        if self._keypoint_model is None:
            print("  WARNING: Court keypoint model not loaded")
            return self._last_transformer

        try:
            result = self._keypoint_model.infer(frame, confidence=self.keypoint_confidence)[0]

            # Try to extract keypoints from the inference result
            keypoints_xy = []
            keypoints_conf = []

            # Handle different result formats
            if hasattr(result, 'predictions') and result.predictions:
                pred = result.predictions[0]
                if hasattr(pred, 'keypoints') and pred.keypoints:
                    for kp in pred.keypoints:
                        keypoints_xy.append([kp.x, kp.y])
                        keypoints_conf.append(kp.confidence)

            if not keypoints_xy:
                # Try supervision KeyPoints format
                try:
                    key_points = sv.KeyPoints.from_inference(result)
                    if key_points.xy is not None and len(key_points.xy) > 0:
                        keypoints_xy = key_points.xy[0].tolist()
                        keypoints_conf = key_points.confidence[0].tolist()
                except Exception:
                    pass

            if not keypoints_xy:
                print(f"  No keypoints detected in frame")
                return self._last_transformer

            self._last_keypoints_xy = np.array(keypoints_xy)
            self._last_keypoints_conf = np.array(keypoints_conf)

            mask = self._last_keypoints_conf > self.anchor_confidence
            num_valid = int(np.sum(mask))
            print(f"  Court keypoints: {num_valid}/{len(keypoints_xy)} above confidence threshold")

            if num_valid >= 4:
                # Use only high-confidence keypoints
                # Make sure vertices array is large enough
                if len(self.vertices) < len(mask):
                    print(f"  WARNING: More keypoints ({len(mask)}) than vertices ({len(self.vertices)})")
                    mask = mask[:len(self.vertices)]

                court_landmarks = self.vertices[mask]
                frame_landmarks = self._last_keypoints_xy[mask]

                self._last_transformer = ViewTransformer(
                    source=frame_landmarks,
                    target=court_landmarks
                )
                print(f"  Homography transformer built successfully with {num_valid} points")
                return self._last_transformer
            else:
                print(f"  Not enough keypoints for homography (need 4, got {num_valid})")
        except Exception as e:
            print(f"  Error building transformer: {e}")

        return self._last_transformer

    def transform_to_court(
        self,
        frame: np.ndarray,
        pixel_positions: np.ndarray
    ) -> Optional[np.ndarray]:
        """Transform pixel positions to court coordinates."""
        if len(pixel_positions) == 0:
            return None

        transformer = self.build_transformer(frame)
        if transformer is None:
            return None

        return transformer.transform_points(points=pixel_positions.astype(np.float32))

    def render(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        frame_shape: Tuple[int, int],
        team_assignments: Dict[int, int] = None,
        team_colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Render tactical court view with player positions.

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
        scale_x = court_w / NBA_COURT_LENGTH
        scale_y = court_h / NBA_COURT_WIDTH

        # Draw each player
        for i, obj_id in enumerate(obj_ids):
            cx_feet, cy_feet = court_xy[i]

            # Convert to pixel coordinates
            px = int((cx_feet + NBA_COURT_LENGTH / 2) * scale_x)
            py = int((NBA_COURT_WIDTH / 2 - cy_feet) * scale_y)

            # Skip if outside court bounds
            if px < 0 or px >= court_w or py < 0 or py >= court_h:
                continue

            # Get team color
            team_id = team_assignments.get(obj_id, 0) if team_assignments else 0
            color = team_colors.get(team_id, (128, 128, 128))

            # Draw filled circle
            radius = 12
            cv2.circle(court_blue, (px, py), radius, color, -1)
            cv2.circle(court_blue, (px, py), radius, (255, 255, 255), 2)

        return court_blue

    def render_with_numbers(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        frame_shape: Tuple[int, int],
        team_assignments: Dict[int, int] = None,
        team_colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Render tactical court view with numbered circles for players."""
        if team_colors is None:
            team_colors = {
                0: (0, 255, 0),
                1: (0, 0, 255),
                -1: (0, 255, 255),
            }

        court_blue = self.get_court_image()

        if self._last_transformer is None or len(player_positions) == 0:
            return court_blue

        obj_ids = list(player_positions.keys())
        positions = np.array([player_positions[oid] for oid in obj_ids], dtype=np.float32)

        court_xy = self._last_transformer.transform_points(points=positions)
        if court_xy is None:
            return court_blue

        court_h, court_w = court_blue.shape[:2]
        scale_x = court_w / NBA_COURT_LENGTH
        scale_y = court_h / NBA_COURT_WIDTH

        for i, obj_id in enumerate(obj_ids):
            cx_feet, cy_feet = court_xy[i]
            px = int((cx_feet + NBA_COURT_LENGTH / 2) * scale_x)
            py = int((NBA_COURT_WIDTH / 2 - cy_feet) * scale_y)

            if px < 0 or px >= court_w or py < 0 or py >= court_h:
                continue

            team_id = team_assignments.get(obj_id, 0) if team_assignments else 0
            color = team_colors.get(team_id, (128, 128, 128))

            radius = 18
            cv2.circle(court_blue, (px, py), radius, color, -1)
            cv2.circle(court_blue, (px, py), radius, (255, 255, 255), 2)

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
        court = draw_court()
        court_hsv = cv2.cvtColor(court, cv2.COLOR_BGR2HSV)
        court_hsv[:, :, 0] = 110
        court_hsv[:, :, 1] = np.clip(court_hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(court_hsv, cv2.COLOR_HSV2BGR)


def create_combined_view(
    frame: np.ndarray,
    tactical_view: np.ndarray,
    position: str = 'bottom-right',
    scale_factor: float = 0.35,
) -> np.ndarray:
    """Overlay tactical view on video frame."""
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

    overlay = combined.copy()
    cv2.rectangle(overlay, (x - 10, y - 10), (x + ntw + 10, y + nth + 10), (20, 20, 20), -1)
    combined = cv2.addWeighted(overlay, 0.9, combined, 0.1, 0)

    cv2.rectangle(combined, (x - 10, y - 10), (x + ntw + 10, y + nth + 10), (80, 80, 80), 2)

    combined[y:y + nth, x:x + ntw] = tactical_resized

    cv2.putText(combined, "TACTICAL VIEW", (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return combined
