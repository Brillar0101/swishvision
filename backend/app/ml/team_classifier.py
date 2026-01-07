"""
Team classification using sports library TeamClassifier.
Uses SigLIP embeddings + UMAP + K-means for accurate team clustering.
"""
import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Optional

# Use the sports library TeamClassifier
from sports import TeamClassifier as SportsTeamClassifier


class TeamClassifier:
    """Classify players into teams using SigLIP embeddings."""
    
    def __init__(self, n_teams: int = 2, device: str = "cpu"):
        self.n_teams = n_teams
        self.device = device
        self._classifier: Optional[SportsTeamClassifier] = None
        self.is_fitted = False
        
        # Team display names
        self.team_names = {
            0: "Team A",
            1: "Team B",
        }
        
        # Team colors for visualization (BGR)
        self.team_colors = {
            0: (0, 255, 0),    # Green for Team A
            1: (0, 0, 255),    # Red for Team B
            -1: (0, 255, 255), # Yellow for referees
        }
    
    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the team classifier on player crops.
        
        Args:
            crops: List of player crop images (center jersey region)
        """
        if len(crops) < self.n_teams:
            print(f"Not enough crops ({len(crops)}) to classify into {self.n_teams} teams")
            return
        
        # Initialize sports library classifier
        self._classifier = SportsTeamClassifier(device=self.device)
        self._classifier.fit(crops)
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
        if not self.is_fitted or self._classifier is None:
            return [0] * len(crops)
        
        return self._classifier.predict(crops)
    
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


def get_player_crops(
    frame: np.ndarray,
    detections: sv.Detections,
    scale_factor: float = 0.4
) -> List[np.ndarray]:
    """
    Extract center crops from player detections for team classification.
    
    Args:
        frame: Video frame
        detections: Player detections
        scale_factor: How much to scale boxes to get center crop (0.4 = 40% of box)
        
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
    scale_factor: float = 0.4
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
        # Fallback if some crops failed
        return np.zeros(len(detections), dtype=int)
    
    teams = classifier.predict(crops)
    return np.array(teams)