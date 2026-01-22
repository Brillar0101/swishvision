"""
Tactical View Module for SwishVision.

Generates smoothed homography-based tactical court view from basketball video.
Uses ycjdo/4 model for player/referee detection and path smoothing.

Homography transformation:
- Source: Keypoints detected in video frame (pixels)
- Target: Court vertices in real-world coordinates (feet)
- Result: Player positions projected to court coordinate system

Team colors (BGR format):
- Team 0: GREEN (0, 255, 0)
- Team 1: RED (0, 0, 255)
- Referees: YELLOW (0, 255, 255)
- Background: BLUE court
"""
import cv2
import pickle
import numpy as np
import supervision as sv
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .player_referee_detector import PlayerRefereeDetector, PLAYER_CLASS_IDS, REFEREE_CLASS_IDS
from .path_smoothing import clean_paths
from .team_classifier import TeamClassifier, get_player_crops

# Try to import from roboflow sports library (basketball branch)
try:
    from sports.basketball import CourtConfiguration, League
    from sports import MeasurementUnit
    SPORTS_LIBRARY_AVAILABLE = True
except ImportError:
    SPORTS_LIBRARY_AVAILABLE = False

# Court keypoint model
COURT_KEYPOINT_MODEL_ID = "basketball-court-detection-2/19"

# NBA Court dimensions in feet
NBA_COURT_LENGTH = 94.0  # Full court length (baseline to baseline)
NBA_COURT_WIDTH = 50.0   # Full court width (sideline to sideline)

# Tactical view dimensions in pixels
TACTICAL_WIDTH = 940
TACTICAL_HEIGHT = 500

# Team colors (BGR format)
TEAM_COLORS = {
    0: (0, 255, 0),    # GREEN for Team 1
    1: (0, 0, 255),    # RED for Team 2
    -1: (0, 255, 255), # YELLOW for referees
}


@dataclass
class BasketballCourtConfiguration:
    """
    Basketball court configuration with 33 keypoints.

    Coordinate system:
    - Origin (0, 0) at top-left corner of the court
    - X: 0 to 94 feet (left baseline to right baseline)
    - Y: 0 to 50 feet (top sideline to bottom sideline)

    The 33 keypoints match the basketball-court-detection-2 model output indices.
    """
    width: float = NBA_COURT_LENGTH   # Court length in feet
    length: float = NBA_COURT_WIDTH   # Court width in feet

    @property
    def vertices(self) -> np.ndarray:
        """
        33 court keypoints in real-world coordinates (feet).
        Order matches basketball-court-detection-2 keypoint model indices.
        """
        # Half court values
        half_length = self.width / 2  # 47 feet
        half_width = self.length / 2  # 25 feet

        # Paint dimensions (16 feet wide = 8 feet each side of basket)
        free_throw_line = 19.0  # 19 feet from baseline

        # Three-point line
        three_point_corner = 22.0  # Corner three distance from sideline
        three_point_arc = 23.75  # Arc radius from basket

        # Basket position from baseline
        basket_from_baseline = 5.25  # 5.25 feet (actually 4 feet + 15 inch rim)

        # Center circle radius
        center_radius = 6.0

        # Free throw circle radius
        ft_circle_radius = 6.0

        # Define all 33 vertices (x, y in feet from top-left origin)
        vertices = np.array([
            # 0-4: Court corners and center
            [half_length, half_width],           # 0: Center court
            [0, 0],                               # 1: Top-left corner
            [0, self.length],                     # 2: Bottom-left corner
            [self.width, 0],                      # 3: Top-right corner
            [self.width, self.length],            # 4: Bottom-right corner

            # 5-9: Left paint
            [0, half_width - 8],                  # 5: Left paint top (baseline)
            [0, half_width + 8],                  # 6: Left paint bottom (baseline)
            [free_throw_line, half_width - 8],   # 7: Left free throw top
            [free_throw_line, half_width + 8],   # 8: Left free throw bottom
            [free_throw_line, half_width],       # 9: Left free throw line center

            # 10-14: Right paint
            [self.width, half_width - 8],        # 10: Right paint top (baseline)
            [self.width, half_width + 8],        # 11: Right paint bottom (baseline)
            [self.width - free_throw_line, half_width - 8],  # 12: Right free throw top
            [self.width - free_throw_line, half_width + 8],  # 13: Right free throw bottom
            [self.width - free_throw_line, half_width],      # 14: Right free throw line center

            # 15-18: Three-point corners
            [three_point_corner, 0],             # 15: Left three-point top corner
            [three_point_corner, self.length],   # 16: Left three-point bottom corner
            [self.width - three_point_corner, 0],        # 17: Right three-point top corner
            [self.width - three_point_corner, self.length],  # 18: Right three-point bottom corner

            # 19-22: Three-point arc intersections with paint extension
            [basket_from_baseline + three_point_arc, half_width - 8],   # 19: Left arc top
            [basket_from_baseline + three_point_arc, half_width + 8],   # 20: Left arc bottom
            [self.width - basket_from_baseline - three_point_arc, half_width - 8],  # 21: Right arc top
            [self.width - basket_from_baseline - three_point_arc, half_width + 8],  # 22: Right arc bottom

            # 23-26: Center line and circle
            [half_length, 0],                    # 23: Center line top
            [half_length, self.length],          # 24: Center line bottom
            [half_length, half_width - center_radius],   # 25: Center circle top
            [half_length, half_width + center_radius],   # 26: Center circle bottom

            # 27-28: Baseline centers
            [0, half_width],                     # 27: Left baseline center
            [self.width, half_width],            # 28: Right baseline center

            # 29-32: Free throw circles
            [free_throw_line, half_width - ft_circle_radius],   # 29: Left FT circle top
            [free_throw_line, half_width + ft_circle_radius],   # 30: Left FT circle bottom
            [self.width - free_throw_line, half_width - ft_circle_radius],  # 31: Right FT circle top
            [self.width - free_throw_line, half_width + ft_circle_radius],  # 32: Right FT circle bottom
        ], dtype=np.float32)

        return vertices

    @property
    def edges(self):
        """Court line edges for drawing (pairs of vertex indices)."""
        return [
            # Court outline
            (1, 3), (3, 4), (4, 2), (2, 1),
            # Center line
            (23, 24),
            # Left paint
            (5, 7), (7, 8), (8, 6),
            # Right paint
            (10, 12), (12, 13), (13, 11),
            # Three-point corners
            (1, 15), (2, 16), (3, 17), (4, 18),
        ]


def get_court_config():
    """
    Get basketball court configuration.

    Uses the roboflow sports library if available (feat/basketball branch),
    otherwise falls back to our custom BasketballCourtConfiguration.

    Install sports library with:
        pip install git+https://github.com/roboflow/sports.git@feat/basketball
    """
    if SPORTS_LIBRARY_AVAILABLE:
        try:
            return CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)
        except Exception:
            pass
    return BasketballCourtConfiguration()


def get_court_vertices() -> np.ndarray:
    """
    Get court keypoint vertices in real-world coordinates (feet).
    These vertices are in the same order as the keypoint model outputs.
    """
    config = get_court_config()
    if SPORTS_LIBRARY_AVAILABLE and hasattr(config, 'vertices'):
        return np.array(config.vertices)
    return config.vertices


def get_court_dimensions() -> Tuple[float, float]:
    """
    Get court dimensions in feet.

    Returns:
        Tuple of (court_length, court_width) in feet.
        - court_length: 94 feet (baseline to baseline)
        - court_width: 50 feet (sideline to sideline)
    """
    # Always use NBA standard dimensions regardless of config source
    return NBA_COURT_LENGTH, NBA_COURT_WIDTH


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


def draw_court(width: int = TACTICAL_WIDTH, height: int = TACTICAL_HEIGHT) -> np.ndarray:
    """
    Draw an NBA basketball court with BLUE background.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Court image (BGR) with blue background
    """
    # Create blue court background
    court = np.ones((height, width, 3), dtype=np.uint8)
    court[:, :] = (180, 100, 50)  # Blue background in BGR

    # Get court dimensions
    court_length, court_width = get_court_dimensions()

    # Scale factors: feet to pixels
    sx = width / court_length
    sy = height / court_width

    line_color = (255, 255, 255)  # White lines

    # Helper to convert feet to pixels
    def to_px(x_feet, y_feet):
        return int(x_feet * sx), int(y_feet * sy)

    # Court center
    center_x = court_length / 2  # 47 feet
    center_y = court_width / 2   # 25 feet

    # Draw court outline
    cv2.rectangle(court, to_px(0, 0), to_px(court_length, court_width), line_color, 2)

    # Draw center line
    cv2.line(court, to_px(center_x, 0), to_px(center_x, court_width), line_color, 2)

    # Paint dimensions
    paint_width = 16.0  # 16 feet wide
    ft_line = 19.0  # Free throw line is 19 feet from baseline

    # Left paint
    cv2.line(court, to_px(0, center_y - paint_width/2), to_px(ft_line, center_y - paint_width/2), line_color, 2)
    cv2.line(court, to_px(ft_line, center_y - paint_width/2), to_px(ft_line, center_y + paint_width/2), line_color, 2)
    cv2.line(court, to_px(0, center_y + paint_width/2), to_px(ft_line, center_y + paint_width/2), line_color, 2)

    # Right paint
    cv2.line(court, to_px(court_length, center_y - paint_width/2), to_px(court_length - ft_line, center_y - paint_width/2), line_color, 2)
    cv2.line(court, to_px(court_length - ft_line, center_y - paint_width/2), to_px(court_length - ft_line, center_y + paint_width/2), line_color, 2)
    cv2.line(court, to_px(court_length, center_y + paint_width/2), to_px(court_length - ft_line, center_y + paint_width/2), line_color, 2)

    # Draw center circle (6 feet radius)
    cv2.circle(court, to_px(center_x, center_y), int(6 * sx), line_color, 2)

    # Free throw circles (6 feet radius)
    cv2.circle(court, to_px(ft_line, center_y), int(6 * sx), line_color, 2)
    cv2.circle(court, to_px(court_length - ft_line, center_y), int(6 * sx), line_color, 2)

    # Basket positions (5.25 feet from baseline, actually 4ft to backboard + 15in to rim center)
    basket_x_left = 5.25
    basket_x_right = court_length - 5.25

    # Three-point line parameters
    three_arc_radius = 23.75  # Arc radius from basket center
    three_corner_dist = 22.0  # Corner three distance from baseline (straight section)
    corner_three_y = 3.0  # Distance from sideline where corner three ends

    # Calculate the angle where the arc meets the corner three line
    # The corner three is 22 feet from sideline, arc center is at basket (5.25 ft from baseline)
    # Arc meets the straight line at y = 3 feet from sideline (top) and y = 47 feet (bottom)
    import math
    # Distance from basket to where arc meets corner = sqrt(radius^2 - (center_y - corner_y)^2)
    arc_to_corner_y = center_y - corner_three_y  # 25 - 3 = 22 feet
    arc_angle = math.degrees(math.asin(arc_to_corner_y / three_arc_radius))  # ~68 degrees

    # Left three-point line
    # Straight sections along sidelines
    cv2.line(court, to_px(0, corner_three_y), to_px(three_corner_dist, corner_three_y), line_color, 2)
    cv2.line(court, to_px(0, court_width - corner_three_y), to_px(three_corner_dist, court_width - corner_three_y), line_color, 2)
    # Arc section
    arc_radius_px = int(three_arc_radius * sx)
    cv2.ellipse(court, to_px(basket_x_left, center_y), (arc_radius_px, int(three_arc_radius * sy)),
                0, -arc_angle, arc_angle, line_color, 2)

    # Right three-point line
    # Straight sections along sidelines
    cv2.line(court, to_px(court_length, corner_three_y), to_px(court_length - three_corner_dist, corner_three_y), line_color, 2)
    cv2.line(court, to_px(court_length, court_width - corner_three_y), to_px(court_length - three_corner_dist, court_width - corner_three_y), line_color, 2)
    # Arc section
    cv2.ellipse(court, to_px(basket_x_right, center_y), (arc_radius_px, int(three_arc_radius * sy)),
                180, -arc_angle, arc_angle, line_color, 2)

    # Restricted area arcs (4 feet radius)
    ra_radius = int(4 * sx)
    cv2.ellipse(court, to_px(basket_x_left, center_y), (ra_radius, int(4 * sy)),
                0, -90, 90, line_color, 2)
    cv2.ellipse(court, to_px(basket_x_right, center_y), (ra_radius, int(4 * sy)),
                180, -90, 90, line_color, 2)

    # Draw baskets (orange circles)
    cv2.circle(court, to_px(basket_x_left, center_y), int(0.75 * sx), (0, 128, 255), -1)
    cv2.circle(court, to_px(basket_x_right, center_y), int(0.75 * sx), (0, 128, 255), -1)

    return court


class TacticalViewProcessor:
    """
    Process video to generate tactical court view with smoothed player positions.

    Uses ycjdo/4 model for detection, homography for coordinate transformation,
    and Savitzky-Golay filter for path smoothing.
    """

    def __init__(
        self,
        keypoint_confidence: float = 0.3,
        anchor_confidence: float = 0.5,
        device: str = "cpu",
    ):
        self.keypoint_confidence = keypoint_confidence
        self.anchor_confidence = anchor_confidence
        self.device = device

        # Initialize components
        self.detector = PlayerRefereeDetector()
        self.team_classifier = TeamClassifier(device=device)
        self.court_vertices = get_court_vertices()

        # Court keypoint model (lazy loaded)
        self._keypoint_model = None

        # Processing state
        self.transformer: Optional[ViewTransformer] = None

    def _load_keypoint_model(self):
        """Lazy load court keypoint model."""
        if self._keypoint_model is not None:
            return

        try:
            from inference import get_model
            self._keypoint_model = get_model(model_id=COURT_KEYPOINT_MODEL_ID)
            print(f"  Court keypoint model loaded: {COURT_KEYPOINT_MODEL_ID}")
        except Exception as e:
            print(f"  Failed to load court keypoint model: {e}")

    def detect_court_keypoints(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect court keypoints with confidence filtering."""
        self._load_keypoint_model()
        if self._keypoint_model is None:
            return None, None

        result = self._keypoint_model.infer(frame, confidence=self.keypoint_confidence)[0]

        # Extract keypoints from result
        keypoints_xy = []
        keypoints_conf = []

        if hasattr(result, 'predictions') and result.predictions:
            pred = result.predictions[0]
            if hasattr(pred, 'keypoints') and pred.keypoints:
                for kp in pred.keypoints:
                    keypoints_xy.append([kp.x, kp.y])
                    keypoints_conf.append(kp.confidence)

        if not keypoints_xy:
            try:
                key_points = sv.KeyPoints.from_inference(result)
                if key_points.xy is not None and len(key_points.xy) > 0:
                    keypoints_xy = key_points.xy[0].tolist()
                    keypoints_conf = key_points.confidence[0].tolist()
            except Exception:
                pass

        if not keypoints_xy:
            return None, None

        return np.array(keypoints_xy), np.array(keypoints_conf)

    def build_transformer(self, frame: np.ndarray) -> bool:
        """Build homography transformer from frame keypoints."""
        xy, conf = self.detect_court_keypoints(frame)

        if xy is None or conf is None:
            return False

        # Filter by confidence
        mask = conf > self.anchor_confidence
        num_valid = int(np.sum(mask))

        if num_valid < 4:
            print(f"  Not enough keypoints for homography (need 4, got {num_valid})")
            return False

        # Ensure we don't exceed vertex count
        if len(mask) > len(self.court_vertices):
            mask = mask[:len(self.court_vertices)]

        frame_points = xy[mask]
        court_points = self.court_vertices[mask]

        self.transformer = ViewTransformer(source=frame_points, target=court_points)
        print(f"  Homography transformer built with {num_valid} points")
        return True

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        resume: bool = False,
    ) -> Dict:
        """
        Process video to generate tactical view.

        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            resume: If True, load from pickle cache; if False, process and save cache

        Returns:
            Dict with output paths and statistics
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cache file paths
        cache_file = output_dir / f"{video_path.stem}_tactical_cache.pkl"
        output_video = output_dir / f"{video_path.stem}_tactical.mp4"

        # Check for resume
        if resume and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Render from cached data
            return self._render_from_cache(
                cache_data,
                str(video_path),
                str(output_video),
                output_dir
            )

        # Process video fresh
        print(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Video: {total_frames} frames, {fps:.1f} fps, {width}x{height}")

        # Build transformer from first few frames
        print("  Building homography transformer...")
        for _ in range(min(30, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            if self.build_transformer(frame):
                break

        if self.transformer is None:
            print("  WARNING: Could not build transformer, using identity")

        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Collect all detections
        print("  Detecting players and collecting positions...")
        all_detections = []
        all_crops = []
        frame_idx = 0

        for _ in tqdm(range(total_frames), desc="Detecting"):
            ret, frame = cap.read()
            if not ret:
                break

            # Detect players and referees
            detections = self.detector.detect_and_track(frame)

            frame_data = {
                'frame_idx': frame_idx,
                'xyxy': detections.xyxy.copy() if len(detections) > 0 else np.array([]),
                'class_id': detections.class_id.copy() if len(detections) > 0 else np.array([]),
                'tracker_id': detections.tracker_id.copy() if detections.tracker_id is not None else np.array([]),
                'confidence': detections.confidence.copy() if len(detections) > 0 else np.array([]),
            }
            all_detections.append(frame_data)

            # Collect crops for team classification (players only, not referees)
            if len(detections) > 0:
                player_mask = np.isin(detections.class_id, PLAYER_CLASS_IDS)
                player_detections = detections[player_mask]
                if len(player_detections) > 0:
                    crops = get_player_crops(frame, player_detections, scale_factor=0.4)
                    all_crops.extend(crops)

            frame_idx += 1

        cap.release()

        # Train team classifier
        print(f"  Training team classifier on {len(all_crops)} crops...")
        if len(all_crops) >= 10:
            # Sample crops for training
            sample_size = min(500, len(all_crops))
            sample_indices = np.random.choice(len(all_crops), sample_size, replace=False)
            sample_crops = [all_crops[i] for i in sample_indices]
            self.team_classifier.fit(sample_crops)

        # Get all unique tracker IDs and build position arrays
        all_tracker_ids = set()
        for det in all_detections:
            if len(det['tracker_id']) > 0:
                all_tracker_ids.update(det['tracker_id'].tolist())

        tracker_id_list = sorted(all_tracker_ids)
        tracker_to_idx = {tid: idx for idx, tid in enumerate(tracker_id_list)}
        n_players = len(tracker_id_list)

        print(f"  Found {n_players} unique tracked objects")

        # Build position array for smoothing: (n_frames, n_players, 2)
        video_xy = np.full((total_frames, n_players, 2), np.nan)
        team_assignments = {}  # tracker_id -> team_id

        # Second pass: classify teams and collect positions
        cap = cv2.VideoCapture(str(video_path))

        for frame_idx, det in enumerate(tqdm(all_detections, desc="Classifying")):
            if len(det['xyxy']) == 0:
                continue

            ret, frame = cap.read()
            if not ret:
                break

            for i in range(len(det['xyxy'])):
                tracker_id = det['tracker_id'][i] if len(det['tracker_id']) > i else i
                class_id = det['class_id'][i]

                # Get foot position (bottom center of bbox)
                x1, _, x2, y2 = det['xyxy'][i]
                foot_x = (x1 + x2) / 2
                foot_y = y2

                # Store position
                if tracker_id in tracker_to_idx:
                    idx = tracker_to_idx[tracker_id]
                    video_xy[frame_idx, idx, :] = [foot_x, foot_y]

                # Classify team (only for players, not referees)
                if tracker_id not in team_assignments:
                    if class_id in REFEREE_CLASS_IDS:
                        team_assignments[tracker_id] = -1  # Referee
                    elif class_id in PLAYER_CLASS_IDS:
                        # Extract crop for this detection
                        box = det['xyxy'][i:i+1]
                        temp_det = sv.Detections(xyxy=box)
                        crops = get_player_crops(frame, temp_det, scale_factor=0.4)
                        if crops and self.team_classifier.is_fitted:
                            team_id = self.team_classifier.predict_single(crops[0])
                            team_assignments[tracker_id] = team_id
                        else:
                            team_assignments[tracker_id] = 0

        cap.release()

        # Apply path smoothing
        print("  Smoothing player paths...")

        # Replace NaN with interpolated values for smoothing
        for p_idx in range(n_players):
            player_xy = video_xy[:, p_idx, :]
            valid_mask = ~np.isnan(player_xy[:, 0])

            if np.sum(valid_mask) < 2:
                continue

            # Interpolate missing values
            valid_indices = np.where(valid_mask)[0]
            for dim in range(2):
                values = player_xy[:, dim]
                invalid_indices = np.where(~valid_mask)[0]
                if len(invalid_indices) > 0:
                    video_xy[invalid_indices, p_idx, dim] = np.interp(
                        invalid_indices, valid_indices, values[valid_indices]
                    )

        # Now apply clean_paths
        smoothed_xy, _ = clean_paths(
            video_xy,
            jump_sigma=3.5,
            min_jump_dist=50.0,  # Pixels
            max_jump_run=18,
            pad_around_runs=2,
            smooth_window=9,
            smooth_poly=2,
        )

        # Transform to court coordinates
        print("  Transforming to court coordinates...")
        court_xy = np.full_like(smoothed_xy, np.nan)

        if self.transformer is not None:
            for frame_idx in range(total_frames):
                positions = smoothed_xy[frame_idx]
                valid_mask = ~np.isnan(positions[:, 0])

                if np.sum(valid_mask) > 0:
                    valid_positions = positions[valid_mask]
                    transformed = self.transformer.transform_points(valid_positions)
                    if transformed is not None:
                        court_xy[frame_idx, valid_mask] = transformed

        # Cache the processed data
        cache_data = {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'tracker_id_list': tracker_id_list,
            'team_assignments': team_assignments,
            'court_xy': court_xy,
            'smoothed_xy': smoothed_xy,
        }

        print(f"  Saving cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        # Render output video
        return self._render_from_cache(
            cache_data,
            str(video_path),
            str(output_video),
            output_dir
        )

    def _render_from_cache(
        self,
        cache_data: Dict,
        video_path: str,
        output_video: str,
        output_dir: Path,
    ) -> Dict:
        """Render tactical view video from cached data."""
        print("  Rendering tactical view video...")

        total_frames = cache_data['total_frames']
        fps = cache_data['fps']
        tracker_id_list = cache_data['tracker_id_list']
        team_assignments = cache_data['team_assignments']
        court_xy = cache_data['court_xy']

        # Tactical view dimensions in pixels
        tactical_width = TACTICAL_WIDTH
        tactical_height = TACTICAL_HEIGHT

        # Get court dimensions in feet
        court_length, court_width = get_court_dimensions()

        # Scale factors: real-world feet to pixels
        scale_x = tactical_width / court_length
        scale_y = tactical_height / court_width

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (tactical_width, tactical_height))

        for frame_idx in tqdm(range(total_frames), desc="Rendering"):
            # Draw court
            court = draw_court(tactical_width, tactical_height)

            # Draw players
            positions = court_xy[frame_idx]

            for p_idx, tracker_id in enumerate(tracker_id_list):
                if np.isnan(positions[p_idx, 0]):
                    continue

                # Positions are in real-world coordinates from homography
                # Convert to pixel coordinates
                real_x, real_y = positions[p_idx]
                px = int(real_x * scale_x)
                py = int(real_y * scale_y)

                # Skip if outside court bounds
                if px < 0 or px >= tactical_width or py < 0 or py >= tactical_height:
                    continue

                # Get team color
                team_id = team_assignments.get(tracker_id, 0)
                color = TEAM_COLORS.get(team_id, (128, 128, 128))

                # Draw filled circle with white border
                radius = 12
                cv2.circle(court, (px, py), radius, color, -1)
                cv2.circle(court, (px, py), radius, (255, 255, 255), 2)

            writer.write(court)

        writer.release()
        print(f"  Output video: {output_video}")

        return {
            'source_video': video_path,
            'output_video': output_video,
            'cache_file': str(output_dir / f"{Path(video_path).stem}_tactical_cache.pkl"),
            'total_frames': total_frames,
            'tracked_objects': len(tracker_id_list),
        }


def process_video(
    video_path: str,
    output_dir: str,
    resume: bool = False,
    device: str = "cpu",
) -> Dict:
    """
    Process video to generate tactical view.

    Args:
        video_path: Path to input video
        output_dir: Output directory
        resume: If True, load from pickle cache

    Returns:
        Dict with output paths
    """
    processor = TacticalViewProcessor(device=device)
    return processor.process_video(video_path, output_dir, resume=resume)


# ============================================================================
# Backwards-compatible classes for player_tracker.py
# ============================================================================

class TacticalView:
    """
    Backwards-compatible TacticalView class for player_tracker.py.
    Converts player positions from video frame to tactical 2D court view.
    """

    def __init__(
        self,
        keypoint_model_id: str = COURT_KEYPOINT_MODEL_ID,
        keypoint_confidence: float = 0.3,
        anchor_confidence: float = 0.5,
    ):
        self.keypoint_model_id = keypoint_model_id
        self.keypoint_confidence = keypoint_confidence
        self.anchor_confidence = anchor_confidence

        self._keypoint_model = None
        self._last_transformer: Optional[ViewTransformer] = None
        self.vertices = get_court_vertices()

        # Get court dimensions for coordinate conversion
        self.court_length, self.court_width = get_court_dimensions()

    def _load_model(self):
        """Lazy load keypoint detection model."""
        if self._keypoint_model is not None:
            return

        try:
            from inference import get_model
            self._keypoint_model = get_model(model_id=self.keypoint_model_id)
        except Exception as e:
            print(f"  Failed to load court keypoint model: {e}")

    def build_transformer(self, frame: np.ndarray) -> Optional[ViewTransformer]:
        """Build homography transformer for a frame."""
        self._load_model()
        if self._keypoint_model is None:
            return self._last_transformer

        try:
            result = self._keypoint_model.infer(frame, confidence=self.keypoint_confidence)[0]

            keypoints_xy = []
            keypoints_conf = []

            if hasattr(result, 'predictions') and result.predictions:
                pred = result.predictions[0]
                if hasattr(pred, 'keypoints') and pred.keypoints:
                    for kp in pred.keypoints:
                        keypoints_xy.append([kp.x, kp.y])
                        keypoints_conf.append(kp.confidence)

            if not keypoints_xy:
                try:
                    key_points = sv.KeyPoints.from_inference(result)
                    if key_points.xy is not None and len(key_points.xy) > 0:
                        keypoints_xy = key_points.xy[0].tolist()
                        keypoints_conf = key_points.confidence[0].tolist()
                except Exception:
                    pass

            if not keypoints_xy:
                return self._last_transformer

            xy = np.array(keypoints_xy)
            conf = np.array(keypoints_conf)

            mask = conf > self.anchor_confidence
            num_valid = int(np.sum(mask))

            if num_valid >= 4:
                if len(mask) > len(self.vertices):
                    mask = mask[:len(self.vertices)]

                frame_points = xy[mask]
                court_points = self.vertices[mask]

                self._last_transformer = ViewTransformer(source=frame_points, target=court_points)
                return self._last_transformer

        except Exception:
            pass

        return self._last_transformer

    def render(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        frame_shape: Tuple[int, int],  # Kept for API compatibility
        team_assignments: Dict[int, int] = None,
        team_colors: Dict[int, Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Render tactical court view with player positions."""
        _ = frame_shape  # Unused but kept for backwards compatibility
        if team_colors is None:
            team_colors = TEAM_COLORS

        court_img = draw_court()

        if self._last_transformer is None or len(player_positions) == 0:
            return court_img

        obj_ids = list(player_positions.keys())
        positions = np.array([player_positions[oid] for oid in obj_ids], dtype=np.float32)

        # Transform to real-world court coordinates
        court_xy = self._last_transformer.transform_points(points=positions)
        if court_xy is None:
            return court_img

        court_h, court_w = court_img.shape[:2]

        # Scale factors: real-world to pixels
        scale_x = court_w / self.court_length
        scale_y = court_h / self.court_width

        for i, obj_id in enumerate(obj_ids):
            # Convert real-world coordinates to pixels
            real_x, real_y = court_xy[i]
            px = int(real_x * scale_x)
            py = int(real_y * scale_y)

            if px < 0 or px >= court_w or py < 0 or py >= court_h:
                continue

            team_id = team_assignments.get(obj_id, 0) if team_assignments else 0
            color = team_colors.get(team_id, (128, 128, 128))

            radius = 12
            cv2.circle(court_img, (px, py), radius, color, -1)
            cv2.circle(court_img, (px, py), radius, (255, 255, 255), 2)

        return court_img


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Tactical Court View")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output-dir", "-o", default="output/tactical_view",
                        help="Output directory")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from cached pickle file")
    parser.add_argument("--device", "-d", default="cpu",
                        help="Device for team classifier (cpu or cuda)")

    args = parser.parse_args()

    results = process_video(
        args.video_path,
        args.output_dir,
        resume=args.resume,
        device=args.device,
    )
    print(f"\nOutput video: {results['output_video']}")
    print(f"Cache file: {results['cache_file']}")
