"""SwishVision — offline computer-vision pipeline for post-game basketball analytics.

See `docs/SPECIFICATION.md` for the v1.0.0 contract.

Public re-exports keep the previous import surface working for callers that did
``from swishvision import PlayerTracker`` directly.
"""

from swishvision.data.team_rosters import TEAM_COLORS, TEAM_ROSTERS, get_player_name
from swishvision.pipeline.court import CourtDetector
from swishvision.pipeline.detection import PlayerRefereeDetector
from swishvision.pipeline.path_smoothing import clean_paths, smooth_tactical_positions
from swishvision.pipeline.tactical import TacticalView, create_combined_view, draw_court
from swishvision.pipeline.team_classifier import TeamClassifier
from swishvision.pipeline.tracker import ConsecutiveValueTracker, PlayerTracker

__version__ = "1.0.0"

__all__ = [
    "ConsecutiveValueTracker",
    "CourtDetector",
    "PlayerRefereeDetector",
    "PlayerTracker",
    "TEAM_COLORS",
    "TEAM_ROSTERS",
    "TacticalView",
    "TeamClassifier",
    "__version__",
    "clean_paths",
    "create_combined_view",
    "draw_court",
    "get_player_name",
    "smooth_tactical_positions",
]
