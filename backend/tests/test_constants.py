"""
Unit tests for constants and configuration values across all modules.
Ensures all constants are properly defined and have valid values.
"""
import pytest
import numpy as np

from app.ml import player_tracker, jersey_detector, team_classifier
from app.ml import tactical_view, path_smoothing, portfolio_generator


class TestPlayerTrackerConstants:
    """Test constants in player_tracker module."""

    def test_class_ids_defined(self):
        """Test that all RF-DETR class IDs are properly defined."""
        assert player_tracker.BALL_CLASS_ID == 0
        assert player_tracker.BALL_IN_BASKET_CLASS_ID == 1
        assert player_tracker.NUMBER_CLASS_ID == 2
        assert player_tracker.REFEREE_CLASS_ID == 8
        assert player_tracker.RIM_CLASS_ID == 9

    def test_player_class_ids_list(self):
        """Test player class IDs list is correct."""
        expected = [3, 4, 5, 6, 7]
        assert player_tracker.PLAYER_CLASS_IDS == expected

    def test_detection_confidence_range(self):
        """Test detection confidence is in valid range."""
        assert 0.0 <= player_tracker.DETECTION_CONFIDENCE <= 1.0

    def test_bytetrack_parameters(self):
        """Test ByteTrack parameters are positive."""
        assert player_tracker.BYTETRACK_LOST_TRACK_BUFFER > 0
        assert 0.0 <= player_tracker.BYTETRACK_TRACK_ACTIVATION_THRESHOLD <= 1.0
        assert player_tracker.BYTETRACK_MINIMUM_CONSECUTIVE_FRAMES > 0

    def test_sam2_parameters(self):
        """Test SAM2 parameters are valid."""
        assert player_tracker.SAM2_CHUNK_SAVE_INTERVAL > 0
        assert "checkpoints" in player_tracker.SAM2_CHECKPOINT
        assert "configs" in player_tracker.SAM2_CONFIG


class TestJerseyDetectorConstants:
    """Test constants in jersey_detector module."""

    def test_ios_threshold_range(self):
        """Test IoS threshold is in valid range."""
        assert 0.0 <= jersey_detector.IOS_THRESHOLD <= 1.0

    def test_padding_values(self):
        """Test padding values are non-negative."""
        assert jersey_detector.NUMBER_PADDING_PX >= 0
        assert jersey_detector.NUMBER_PADDING_PY >= 0

    def test_consecutive_validation_frames(self):
        """Test consecutive validation is positive."""
        assert jersey_detector.CONSECUTIVE_VALIDATION_FRAMES > 0

    def test_ocr_image_size(self):
        """Test OCR image size is valid."""
        width, height = jersey_detector.OCR_IMAGE_SIZE
        assert width > 0 and height > 0
        assert isinstance(width, int) and isinstance(height, int)

    def test_ocr_prompt_defined(self):
        """Test OCR prompt is defined."""
        assert len(jersey_detector.OCR_PROMPT) > 0


class TestTeamClassifierConstants:
    """Test constants in team_classifier module."""

    def test_siglip_model_name(self):
        """Test SigLIP model name is valid."""
        assert "siglip" in team_classifier.SIGLIP_MODEL_NAME.lower()
        assert "/" in team_classifier.SIGLIP_MODEL_NAME  # Format: org/model

    def test_embedding_dimension(self):
        """Test embedding dimension is positive."""
        assert team_classifier.SIGLIP_EMBEDDING_DIM == 768

    def test_kmeans_parameters(self):
        """Test K-means parameters are valid."""
        assert team_classifier.DEFAULT_N_TEAMS == 2
        assert team_classifier.KMEANS_RANDOM_STATE >= 0
        assert team_classifier.KMEANS_N_INIT > 0

    def test_crop_scale_factor(self):
        """Test crop scale factor is in valid range."""
        assert 0.0 < team_classifier.DEFAULT_CROP_SCALE_FACTOR < 1.0

    def test_team_colors_defined(self):
        """Test team colors are properly defined."""
        colors = team_classifier.DEFAULT_TEAM_COLORS
        assert 0 in colors  # Team A
        assert 1 in colors  # Team B
        assert -1 in colors  # Referees
        # Check BGR format (3 values per color)
        for color in colors.values():
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


class TestTacticalViewConstants:
    """Test constants in tactical_view module."""

    def test_homography_min_landmarks(self):
        """Test minimum landmarks for homography is valid."""
        assert tactical_view.MIN_LANDMARKS_FOR_HOMOGRAPHY >= 4

    def test_court_dimensions(self):
        """Test court dimensions are positive."""
        assert tactical_view.COURT_DEFAULT_WIDTH > 0
        assert tactical_view.COURT_DEFAULT_HEIGHT > 0

    def test_court_colors_valid(self):
        """Test court colors are valid BGR tuples."""
        bg_color = tactical_view.COURT_FALLBACK_BG_COLOR
        outline_color = tactical_view.COURT_OUTLINE_COLOR
        assert len(bg_color) == 3
        assert len(outline_color) == 3
        assert all(0 <= c <= 255 for c in bg_color)
        assert all(0 <= c <= 255 for c in outline_color)

    def test_overlay_parameters(self):
        """Test tactical overlay parameters are valid."""
        assert 0.0 < tactical_view.TACTICAL_OVERLAY_SCALE < 1.0
        assert tactical_view.TACTICAL_OVERLAY_MARGIN > 0
        assert tactical_view.TACTICAL_OVERLAY_BORDER_THICKNESS > 0


class TestPathSmoothingConstants:
    """Test constants in path_smoothing module."""

    def test_jump_detection_parameters(self):
        """Test jump detection parameters are positive."""
        assert path_smoothing.DEFAULT_JUMP_SIGMA > 0
        assert path_smoothing.DEFAULT_MIN_JUMP_DIST > 0
        assert path_smoothing.MAD_TO_STD_SCALE > 0

    def test_run_expansion_parameters(self):
        """Test run expansion parameters are non-negative."""
        assert path_smoothing.DEFAULT_MAX_JUMP_RUN >= 0
        assert path_smoothing.DEFAULT_PAD_AROUND_RUNS >= 0

    def test_smoothing_window_parameters(self):
        """Test Savitzky-Golay window parameters."""
        assert path_smoothing.DEFAULT_SMOOTH_WINDOW > 0
        assert path_smoothing.DEFAULT_SMOOTH_WINDOW % 2 == 1  # Must be odd
        assert path_smoothing.DEFAULT_SMOOTH_POLY > 0
        assert path_smoothing.DEFAULT_SMOOTH_POLY < path_smoothing.DEFAULT_SMOOTH_WINDOW

    def test_tactical_window_size(self):
        """Test tactical window size is positive."""
        assert path_smoothing.DEFAULT_TACTICAL_WINDOW > 0


class TestPortfolioGeneratorConstants:
    """Test constants in portfolio_generator module."""

    def test_color_count(self):
        """Test distinct colors count is positive."""
        assert portfolio_generator.DISTINCT_COLORS_COUNT > 0

    def test_thickness_values(self):
        """Test thickness values are positive."""
        assert portfolio_generator.BOX_ANNOTATOR_THICKNESS > 0
        assert portfolio_generator.BOX_THICKNESS > 0
        assert portfolio_generator.CONTOUR_THICKNESS > 0

    def test_alpha_values(self):
        """Test alpha values are in valid range."""
        assert 0.0 <= portfolio_generator.MASK_OVERLAY_ALPHA <= 1.0
        assert 0.0 <= portfolio_generator.TEAM_OVERLAY_ALPHA <= 1.0

    def test_video_dimensions(self):
        """Test tactical video dimensions are positive."""
        assert portfolio_generator.TACTICAL_VIDEO_WIDTH > 0
        assert portfolio_generator.TACTICAL_VIDEO_HEIGHT > 0
        # Check 16:9 aspect ratio
        ratio = portfolio_generator.TACTICAL_VIDEO_WIDTH / portfolio_generator.TACTICAL_VIDEO_HEIGHT
        assert abs(ratio - 16/9) < 0.01

    def test_label_offsets(self):
        """Test label offset values are non-negative."""
        assert portfolio_generator.LABEL_OFFSET_Y >= 0
        assert portfolio_generator.LABEL_OFFSET_Y_SMALL >= 0

    def test_hue_max(self):
        """Test HUE_MAX is correct for OpenCV."""
        assert portfolio_generator.HUE_MAX == 180  # OpenCV uses 0-180 for hue


class TestConstantsConsistency:
    """Test consistency between constants across modules."""

    def test_player_class_ids_consistency(self):
        """Test player class IDs are consistent across modules."""
        # All modules should use the same player class IDs
        tracker_ids = set(player_tracker.PLAYER_CLASS_IDS)
        tactical_ids = set(tactical_view.PLAYER_CLASS_IDS)
        assert tracker_ids == tactical_ids

    def test_team_colors_consistency(self):
        """Test team colors format is consistent."""
        # All color dictionaries should have teams 0, 1, and -1
        tactical_colors = tactical_view.TEAM_COLORS
        team_colors = team_classifier.DEFAULT_TEAM_COLORS

        assert set(tactical_colors.keys()) == {0, 1, -1}
        assert set(team_colors.keys()) == {0, 1, -1}

    def test_confidence_thresholds_valid(self):
        """Test all confidence thresholds are in valid range."""
        # Collect all confidence thresholds
        confidences = [
            player_tracker.DETECTION_CONFIDENCE,
            jersey_detector.IOS_THRESHOLD,
            tactical_view.KEYPOINT_CONFIDENCE,
            tactical_view.ANCHOR_CONFIDENCE,
        ]

        for conf in confidences:
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
