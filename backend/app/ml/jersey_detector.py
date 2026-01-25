"""
Jersey Number Detection using Roboflow's approach.
Based on the Roboflow basketball AI notebook.

Uses:
- RF-DETR for player and number detection
- SmolVLM2 for jersey number OCR
- Mask IoS matching for number-to-player association
- Consecutive frame validation for stable predictions
"""
import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import supervision as sv

from app.ml.team_rosters import TEAM_ROSTERS, get_player_name

# ============================================================================
# CONSTANTS
# ============================================================================

# Model IDs
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
NUMBER_OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"

# Detection parameters
DETECTION_CONFIDENCE = 0.4
DETECTION_IOU_THRESHOLD = 0.9

# Number-to-player matching
IOS_THRESHOLD = 0.9  # Intersection over Smaller threshold
NUMBER_PADDING_PX = 10  # Padding around number boxes for OCR
NUMBER_PADDING_PY = 10

# Validation
CONSECUTIVE_VALIDATION_FRAMES = 3  # Require N consecutive reads to validate
NUMBER_CLASS_ID = 2  # Class ID for 'number' in RF-DETR model

# OCR
OCR_PROMPT = "Read the number."
OCR_IMAGE_SIZE = (224, 224)  # Resize crops to this size for OCR


class ConsecutiveValueTracker:
    """
    Tracks values over time and validates them based on consecutive occurrences.
    A value is only confirmed after appearing n_consecutive times in a row.
    """

    def __init__(self, n_consecutive: int = 3):
        self.n_consecutive = n_consecutive
        self._current_values: Dict[int, str] = {}
        self._consecutive_counts: Dict[int, int] = defaultdict(int)
        self._validated_values: Dict[int, str] = {}

    def update(self, tracker_ids: List[int], values: List[str]) -> None:
        """Update tracker with new observations."""
        for tracker_id, value in zip(tracker_ids, values):
            if tracker_id in self._validated_values:
                continue

            if self._current_values.get(tracker_id) == value:
                self._consecutive_counts[tracker_id] += 1
            else:
                self._current_values[tracker_id] = value
                self._consecutive_counts[tracker_id] = 1

            if self._consecutive_counts[tracker_id] >= self.n_consecutive:
                self._validated_values[tracker_id] = value

    def get_validated(self, tracker_ids: List[int]) -> List[Optional[str]]:
        """Get validated values for given tracker IDs."""
        return [self._validated_values.get(tid) for tid in tracker_ids]

    def get_all_validated(self) -> Dict[int, str]:
        """Get all validated values."""
        return self._validated_values.copy()

    def reset(self) -> None:
        """Reset all tracking state."""
        self._current_values.clear()
        self._consecutive_counts.clear()
        self._validated_values.clear()


class JerseyDetector:
    """
    Detects and recognizes jersey numbers using Roboflow models.

    Uses:
    - RF-DETR for detecting 'number' bounding boxes
    - SmolVLM2 for reading the number text
    - Mask IoS matching to associate numbers with players
    """

    def __init__(
        self,
        detection_model_id: str = PLAYER_DETECTION_MODEL_ID,
        ocr_model_id: str = NUMBER_OCR_MODEL_ID,
        detection_confidence: float = DETECTION_CONFIDENCE,
        iou_threshold: float = DETECTION_IOU_THRESHOLD,
        n_consecutive: int = CONSECUTIVE_VALIDATION_FRAMES,
        ocr_interval: int = 5,
    ):
        self.detection_model_id = detection_model_id
        self.ocr_model_id = ocr_model_id
        self.detection_confidence = detection_confidence
        self.iou_threshold = iou_threshold
        self.n_consecutive = n_consecutive
        self.ocr_interval = ocr_interval

        self.number_validator = ConsecutiveValueTracker(n_consecutive=n_consecutive)
        self.team_validator = ConsecutiveValueTracker(n_consecutive=1)

        self._detection_model = None
        self._ocr_model = None
        self._initialized = False

        # Class IDs from the detection model
        self.NUMBER_CLASS_ID = 2
        self.PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]

    def _ensure_initialized(self):
        """Lazy initialization of models."""
        if self._initialized:
            return

        try:
            from inference import get_model

            print("Loading jersey detection models...")
            self._detection_model = get_model(model_id=self.detection_model_id)
            self._ocr_model = get_model(model_id=self.ocr_model_id)
            self._initialized = True
            print("Jersey detection models loaded successfully")
        except Exception as e:
            print(f"Failed to load jersey detection models: {e}")
            print("Jersey detection will be disabled")
            self._initialized = False

    def detect_numbers(self, frame: np.ndarray) -> sv.Detections:
        """Detect number bounding boxes in a frame."""
        self._ensure_initialized()
        if not self._initialized:
            return sv.Detections.empty()

        result = self._detection_model.infer(
            frame,
            confidence=self.detection_confidence,
            iou_threshold=self.iou_threshold
        )[0]
        detections = sv.Detections.from_inference(result)
        return detections[detections.class_id == self.NUMBER_CLASS_ID]

    def recognize_number(self, crop: np.ndarray) -> Optional[str]:
        """Recognize jersey number from a cropped image."""
        self._ensure_initialized()
        if not self._initialized:
            return None

        try:
            crop_resized = sv.resize_image(crop, resolution_wh=(224, 224))
            result = self._ocr_model.predict(crop_resized, "Read the number.")[0]
            if result and result.strip().isdigit():
                return result.strip()
        except Exception as e:
            pass
        return None

    def match_numbers_to_players(
        self,
        frame: np.ndarray,
        player_masks: np.ndarray,
        player_tracker_ids: np.ndarray,
    ) -> Dict[int, str]:
        """
        Match detected numbers to player masks using IoS.

        Args:
            frame: Video frame
            player_masks: Array of player segmentation masks
            player_tracker_ids: Array of tracker IDs for each player

        Returns:
            Dict mapping tracker_id -> jersey number
        """
        self._ensure_initialized()
        if not self._initialized:
            return {}

        frame_h, frame_w = frame.shape[:2]

        # Detect number boxes
        number_detections = self.detect_numbers(frame)
        if len(number_detections) == 0:
            return {}

        # Convert number boxes to masks for IoS calculation
        number_masks = sv.xyxy_to_mask(
            boxes=number_detections.xyxy,
            resolution_wh=(frame_w, frame_h)
        )

        # Calculate IoS between player masks and number masks
        iou_matrix = sv.mask_iou_batch(
            masks_true=player_masks,
            masks_detection=number_masks,
            overlap_metric=sv.OverlapMetric.IOS
        )

        # Find matches above threshold
        matches = {}
        for player_idx in range(len(player_masks)):
            for number_idx in range(len(number_masks)):
                if iou_matrix[player_idx, number_idx] >= 0.9:
                    # Crop the number region
                    box = number_detections.xyxy[number_idx]
                    padded_box = sv.pad_boxes(
                        xyxy=np.array([box]),
                        px=10, py=10
                    )[0]
                    clipped_box = sv.clip_boxes(
                        xyxy=np.array([padded_box]),
                        resolution_wh=(frame_w, frame_h)
                    )[0]

                    crop = sv.crop_image(frame, clipped_box)
                    number = self.recognize_number(crop)

                    if number:
                        tracker_id = player_tracker_ids[player_idx]
                        matches[int(tracker_id)] = number
                        break

        return matches

    def update_and_validate(
        self,
        frame: np.ndarray,
        frame_idx: int,
        player_masks: np.ndarray,
        player_tracker_ids: np.ndarray,
    ) -> Dict[int, str]:
        """
        Update jersey number tracking for a frame.

        Args:
            frame: Video frame
            frame_idx: Frame index
            player_masks: Array of player segmentation masks
            player_tracker_ids: Array of tracker IDs

        Returns:
            Dict of validated jersey numbers (tracker_id -> number)
        """
        # Only run OCR at specified intervals
        if frame_idx % self.ocr_interval == 0:
            matches = self.match_numbers_to_players(
                frame, player_masks, player_tracker_ids
            )
            if matches:
                self.number_validator.update(
                    tracker_ids=list(matches.keys()),
                    values=list(matches.values())
                )

        return self.number_validator.get_all_validated()

    def get_player_label(
        self,
        tracker_id: int,
        team_name: Optional[str] = None,
    ) -> str:
        """
        Get display label for a player.

        Args:
            tracker_id: Player's tracker ID
            team_name: Name of the team (for roster lookup)

        Returns:
            Label string like "#23 Nesmith" or "#5" or "Player 5"
        """
        validated = self.number_validator.get_all_validated()
        jersey_number = validated.get(tracker_id)

        if jersey_number:
            player_name = None
            if team_name and team_name in TEAM_ROSTERS:
                player_name = get_player_name(team_name, jersey_number)

            if player_name:
                return f"#{jersey_number} {player_name}"
            else:
                return f"#{jersey_number}"
        else:
            return f"Player {tracker_id}"

    def reset(self):
        """Reset all tracking state."""
        self.number_validator.reset()
        self.team_validator.reset()
