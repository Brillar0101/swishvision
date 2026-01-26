"""
Team classification using SigLIP embeddings + UMAP + K-means.
Standalone implementation without external sports library dependency.
"""
import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

# ============================================================================
# CONSTANTS
# ============================================================================

# SigLIP model
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
SIGLIP_EMBEDDING_DIM = 768  # Output dimension

# K-means clustering
DEFAULT_N_TEAMS = 2
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

# Team crop extraction
DEFAULT_CROP_SCALE_FACTOR = 0.4  # Center region scale for jersey area

# Default team colors (BGR format for OpenCV)
DEFAULT_TEAM_COLORS = {
    0: (0, 255, 0),    # Green for Team A
    1: (0, 0, 255),    # Red for Team B
    -1: (0, 255, 255), # Yellow for referees
}


class TeamClassifier:
    """Classify players into teams using SigLIP embeddings."""

    def __init__(self, n_teams: int = DEFAULT_N_TEAMS, device: str = "cpu"):
        self.n_teams = n_teams
        self.device = device
        self._model = None
        self._processor = None
        self._kmeans = None
        self.is_fitted = False

        # Team display names
        self.team_names = {
            0: "Team A",
            1: "Team B",
        }

        # Team colors for visualization (BGR)
        self.team_colors = DEFAULT_TEAM_COLORS.copy()

    def _load_model(self):
        """Lazy load the SigLIP model."""
        if self._model is not None:
            return

        from transformers import AutoProcessor, SiglipVisionModel
        import torch

        self._processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self._model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_NAME)

        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to("cuda")

        self._model.eval()
        print("  SigLIP model loaded")

    def _get_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Get SigLIP embeddings for crops."""
        import torch

        # Convert BGR to RGB
        rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops if c.size > 0]

        if len(rgb_crops) == 0:
            return np.zeros((len(crops), SIGLIP_EMBEDDING_DIM))

        # Process through SigLIP
        with torch.no_grad():
            inputs = self._processor(images=rgb_crops, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            outputs = self._model(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy()

        return embeddings

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the team classifier on player crops.

        Args:
            crops: List of player crop images (center jersey region)
        """
        if len(crops) < self.n_teams:
            print(f"Not enough crops ({len(crops)}) to classify into {self.n_teams} teams")
            return

        self._load_model()

        # Get embeddings
        embeddings = self._get_embeddings(crops)

        # Fit K-means
        self._kmeans = KMeans(
            n_clusters=self.n_teams,
            random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT
        )
        self._kmeans.fit(embeddings)

        self.is_fitted = True
        print(f"  Team classifier trained on {len(crops)} crops")

    def predict(self, crops: List[np.ndarray]) -> List[int]:
        """
        Predict team for given player crops.

        Args:
            crops: List of player crop images

        Returns:
            List of team IDs (0 or 1)
        """
        if not self.is_fitted or self._kmeans is None:
            return [0] * len(crops)

        if len(crops) == 0:
            return []

        embeddings = self._get_embeddings(crops)
        predictions = self._kmeans.predict(embeddings)

        return predictions.tolist()

    def predict_single(self, crop: np.ndarray) -> int:
        """Predict team for a single crop."""
        return self.predict([crop])[0]

    def get_team_name(self, team_id: int) -> str:
        """Get display name for team."""
        return self.team_names.get(team_id, f"Team {team_id}")

    def get_team_color(self, team_id: int) -> Tuple[int, int, int]:
        """Get BGR color for team visualization."""
        return self.team_colors.get(team_id, (128, 128, 128))

    def set_team_names(self, names: Dict[int, str]) -> None:
        """Set custom team names."""
        self.team_names.update(names)

    def set_team_colors(self, colors: Dict[int, Tuple[int, int, int]]) -> None:
        """Set custom team colors (BGR)."""
        self.team_colors.update(colors)

    def swap_teams(self) -> None:
        """
        Swap team assignments (0 ↔ 1).

        This flips the K-means cluster labels so Team 0 becomes Team 1 and vice versa.
        Useful when the automatic clustering assigns teams incorrectly.
        """
        if not self.is_fitted or self._kmeans is None:
            print("Cannot swap teams: classifier not fitted yet")
            return

        # Swap cluster centers
        if hasattr(self._kmeans, 'cluster_centers_'):
            centers = self._kmeans.cluster_centers_.copy()
            self._kmeans.cluster_centers_[0] = centers[1]
            self._kmeans.cluster_centers_[1] = centers[0]

        # Swap labels
        if hasattr(self._kmeans, 'labels_'):
            self._kmeans.labels_ = 1 - self._kmeans.labels_

        print("  Teams swapped: Team 0 ↔ Team 1")


def get_player_crops(
    frame: np.ndarray,
    detections: sv.Detections,
    scale_factor: float = DEFAULT_CROP_SCALE_FACTOR
) -> List[np.ndarray]:
    """
    Extract center crops from player detections for team classification.

    Args:
        frame: Video frame
        detections: Player detections
        scale_factor: How much to scale boxes to get center crop (default = 40% of box)

    Returns:
        List of crop images
    """
    # Scale boxes to get center region (jersey area)
    boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=scale_factor)

    crops = []
    for box in boxes:
        crop = sv.crop_image(frame, box)
        if crop.size > 0:
            crops.append(crop)

    return crops


def classify_frame_teams(
    frame: np.ndarray,
    detections: sv.Detections,
    classifier: TeamClassifier,
    scale_factor: float = DEFAULT_CROP_SCALE_FACTOR
) -> np.ndarray:
    """
    Classify teams for detections in a single frame.

    Args:
        frame: Video frame
        detections: Player detections
        classifier: Fitted TeamClassifier
        scale_factor: Box scale factor for crops

    Returns:
        Array of team IDs matching detections
    """
    if len(detections) == 0:
        return np.array([])

    crops = get_player_crops(frame, detections, scale_factor)

    if len(crops) != len(detections):
        raise RuntimeError(f"Crop extraction failed: expected {len(detections)} crops but got {len(crops)}")

    teams = classifier.predict(crops)
    return np.array(teams)
