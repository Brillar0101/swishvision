"""Single source of truth for tunable pipeline parameters.

Per project convention (see ``docs/SPECIFICATION.md`` §6.4 NFR-30 and the
project memory entry "All tunable values must be module-level constants"):
**every tunable lives here as one named constant**. Function bodies and
function-default arguments reference these constants, never inline numerics.
Updating a value in this file changes the whole pipeline.

When introducing a new tunable, prefer adding it here over a per-module
constant -- pipeline-level coherence is more important than locality.

Naming convention: ``UPPER_SNAKE_CASE``, prefixed by stage when ambiguous
(``DETECTION_*``, ``BYTETRACK_*``, ``SAM2_*``, ``TEAM_*``, ``JERSEY_*``,
``COURT_*``, ``SMOOTH_*``).
"""
from __future__ import annotations

from pathlib import Path

# ============================================================================
# Filesystem layout
# ============================================================================
# Anchor every relative path against the repository root so the pipeline runs
# the same regardless of the caller's CWD.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
TEST_VIDEOS_DIR: Path = ASSETS_DIR / "test_videos"
CHECKPOINTS_DIR: Path = ASSETS_DIR / "checkpoints"
MODELS_DIR: Path = ASSETS_DIR / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

DEFAULT_TEST_VIDEO: Path = TEST_VIDEOS_DIR / "test_game.mp4"
DEFAULT_RFDETR_WEIGHTS: Path = MODELS_DIR / "rf-detr-base.pth"
DEFAULT_SAM2_LARGE_CHECKPOINT: Path = CHECKPOINTS_DIR / "sam2.1_hiera_large.pt"
DEFAULT_SAM2_SMALL_CHECKPOINT: Path = CHECKPOINTS_DIR / "sam2.1_hiera_small.pt"

DEFAULT_SAM2_CONFIG_LARGE: str = "sam2.1_hiera_l"
DEFAULT_SAM2_CONFIG_SMALL: str = "sam2.1_hiera_s"

# ============================================================================
# Roboflow Universe model IDs (see docs/SPECIFICATION.md §8.1)
# ============================================================================
PLAYER_DETECTION_MODEL_ID: str = "basketball-player-detection-3-ycjdo/4"
JERSEY_OCR_MODEL_ID: str = "basketball-jersey-numbers-ocr/3"
COURT_KEYPOINT_MODEL_ID: str = "basketball-court-detection-2/14"

# ============================================================================
# Detection (RF-DETR + class IDs)
# ============================================================================
PLAYER_CLASS_IDS: tuple[int, ...] = (3, 4, 5, 6, 7)
NUMBER_CLASS_ID: int = 2
REFEREE_CLASS_IDS: tuple[int, ...] = (8,)

DETECTION_CONFIDENCE: float = 0.4
DETECTION_NMS_IOU_THRESHOLD: float = 0.9  # collapses RF-DETR / jersey overlap (FR-32)

# ============================================================================
# Tracking (ByteTrack)
# ============================================================================
BYTETRACK_TRACK_ACTIVATION_THRESHOLD: float = 0.15
BYTETRACK_LOST_TRACK_BUFFER: int = 600  # ≈ 20 s at 30 fps (FR-12)
BYTETRACK_MINIMUM_MATCHING_THRESHOLD: float = 0.5
BYTETRACK_MINIMUM_CONSECUTIVE_FRAMES: int = 1  # 3 in some legacy paths -- track in spec

# ============================================================================
# Pipeline orchestration
# ============================================================================
DEFAULT_MAX_TOTAL_OBJECTS: int = 20  # cap on tracked objects (FR-13)
DEFAULT_KEYFRAME_INTERVAL: int = 30  # frames between SAM2 keyframe prompts
DEFAULT_BOX_MATCH_IOU_THRESHOLD: float = 0.3  # mid-stream new-prompt match
DEFAULT_NUM_SAMPLE_FRAMES: int = 3  # portfolio sample count

# ============================================================================
# SAM2 segmentation
# ============================================================================
SAM2_MASK_FILTER_RELATIVE_DISTANCE: float = 0.03
SAM2_FORCE_FP32: bool = True  # avoids bfloat16 / float32 clash on H100/H200 (FR-15)

# ============================================================================
# Team classification (SigLIP + K-means)
# ============================================================================
TEAM_KMEANS_RANDOM_STATE: int = 42
TEAM_KMEANS_N_INIT: int = 10
TEAM_CROP_SCALE: float = 0.4  # centre-crop scale on player box (FR-20)
TEAM_TRAINING_SAMPLE_FPS: float = 1.0

# ============================================================================
# Jersey number OCR
# ============================================================================
JERSEY_RECOGNITION_INTERVAL: int = 5  # process every Nth frame (FR-30)
JERSEY_DETECTION_IOU_THRESHOLD: float = 0.9  # IoS jersey↔player (FR-32)
JERSEY_CONSECUTIVE_VALIDATION_FRAMES: int = 3  # FR-33

# ============================================================================
# Court detection / homography
# ============================================================================
COURT_MIN_KEYPOINTS: int = 8  # below this, tactical render falls back

# ============================================================================
# Path smoothing (FR-43)
# ============================================================================
SMOOTH_JUMP_SIGMA: float = 3.5
SMOOTH_MIN_JUMP_DIST: float = 0.6
SMOOTH_MAX_JUMP_RUN: int = 18
SMOOTH_PAD_AROUND_RUNS: int = 2
SMOOTH_WINDOW: int = 9  # Savitzky-Golay window
SMOOTH_POLY: int = 2  # Savitzky-Golay polynomial order
SMOOTH_TACTICAL_WINDOW: int = 5

# ============================================================================
# Rendering
# ============================================================================
RENDER_TEAM_OVERLAY_ALPHA: float = 0.4
RENDER_BOX_THICKNESS: int = 3
RENDER_TITLE_BAR_HEIGHT: int = 70

# Reproducible random sampling for the smoke-test runner
SAMPLE_SEED: int = 42
SAMPLE_EDGE_TRIM: float = 0.10  # avoid first/last 10 % of frames
SMOKE_NUM_RANDOM_FRAMES: int = 5
SMOKE_MAX_SECONDS: float | None = None  # None = full clip
