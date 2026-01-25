"""
Unit tests for portfolio generator helper functions.
Tests the refactored rendering utilities.
"""
import pytest
import numpy as np
import cv2

from app.ml.portfolio_generator import (
    mask_to_box,
    draw_mask_overlay,
    draw_player_annotation,
    TEAM_OVERLAY_ALPHA
)


class TestMaskToBox:
    """Test mask_to_box function."""

    def test_valid_mask(self):
        """Test conversion of valid mask to bounding box."""
        # Create a 100x100 mask with a 20x30 rectangle
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 30:60] = True

        box = mask_to_box(mask)

        assert box is not None
        assert len(box) == 4
        assert box[0] == 30  # x_min
        assert box[1] == 40  # y_min
        assert box[2] == 59  # x_max
        assert box[3] == 59  # y_max

    def test_empty_mask(self):
        """Test that empty mask returns None."""
        mask = np.zeros((100, 100), dtype=bool)
        box = mask_to_box(mask)
        assert box is None

    def test_3d_mask(self):
        """Test that 3D mask is properly squeezed."""
        # Create 3D mask (height, width, 1)
        mask = np.zeros((100, 100, 1), dtype=bool)
        mask[50:70, 40:80, 0] = True

        box = mask_to_box(mask)

        assert box is not None
        assert box[0] == 40
        assert box[1] == 50
        assert box[2] == 79
        assert box[3] == 69

    def test_single_pixel_mask(self):
        """Test mask with single pixel."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 60] = True

        box = mask_to_box(mask)

        assert box is not None
        assert box[0] == box[2] == 60
        assert box[1] == box[3] == 50


class TestDrawMaskOverlay:
    """Test draw_mask_overlay function."""

    def test_overlay_with_alpha(self):
        """Test that overlay is applied with correct alpha."""
        # Create test frame and mask
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100  # Gray frame
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 30:70] = True

        color = (0, 255, 0)  # Green
        result = draw_mask_overlay(frame, mask, color, alpha=0.5)

        # Check that result has correct shape
        assert result.shape == frame.shape

        # Check that masked area has green tint
        masked_pixel = result[50, 50]
        assert masked_pixel[1] > masked_pixel[0]  # More green than blue
        assert masked_pixel[1] > masked_pixel[2]  # More green than red

        # Check that non-masked area is unchanged
        non_masked_pixel = result[10, 10]
        assert np.array_equal(non_masked_pixel, [100, 100, 100])

    def test_overlay_preserves_dtype(self):
        """Test that overlay preserves frame dtype."""
        frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.ones((50, 50), dtype=bool)
        color = (255, 0, 0)

        result = draw_mask_overlay(frame, mask, color)

        assert result.dtype == np.uint8

    def test_overlay_with_zero_alpha(self):
        """Test overlay with alpha=0 (no overlay)."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        mask = np.ones((100, 100), dtype=bool)
        color = (0, 255, 0)

        result = draw_mask_overlay(frame, mask, color, alpha=0.0)

        # Frame should be almost unchanged (within rounding)
        assert np.allclose(result, frame, atol=1)


class TestDrawPlayerAnnotation:
    """Test draw_player_annotation function."""

    def test_full_annotation(self):
        """Test drawing complete annotation (mask, box, label)."""
        frame = np.ones((200, 300, 3), dtype=np.uint8) * 128
        mask = np.zeros((200, 300), dtype=bool)
        mask[80:120, 100:200] = True

        color = (0, 255, 0)
        label = "Test Player"

        result = draw_player_annotation(frame, mask, color, label)

        # Check that result has correct shape
        assert result.shape == frame.shape

        # Check that some pixels have changed (annotation was drawn)
        assert not np.array_equal(result, frame)

    def test_annotation_without_box(self):
        """Test drawing annotation without bounding box."""
        frame = np.ones((200, 300, 3), dtype=np.uint8) * 128
        mask = np.zeros((200, 300), dtype=bool)
        mask[80:120, 100:200] = True

        color = (255, 0, 0)
        label = "No Box"

        result = draw_player_annotation(frame, mask, color, label, draw_box=False)

        # Result should still be different (mask overlay applied)
        assert not np.array_equal(result, frame)

    def test_annotation_with_custom_alpha(self):
        """Test annotation with custom transparency."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 30:70] = True

        color = (0, 0, 255)
        label = "Alpha Test"

        # Test with different alphas
        result_low = draw_player_annotation(frame, mask, color, label, alpha=0.2)
        result_high = draw_player_annotation(frame, mask, color, label, alpha=0.8)

        # High alpha should have more pronounced color
        pixel_low = result_low[50, 50, 2]  # Red channel
        pixel_high = result_high[50, 50, 2]  # Red channel

        assert pixel_high > pixel_low

    def test_annotation_with_empty_mask(self):
        """Test that empty mask is handled gracefully."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.zeros((100, 100), dtype=bool)

        color = (0, 255, 0)
        label = "Empty Mask"

        # Should not raise an exception
        result = draw_player_annotation(frame, mask, color, label)

        # Frame should be almost unchanged
        assert np.allclose(result, frame, atol=1)


class TestHelperFunctionsIntegration:
    """Integration tests for helper functions working together."""

    def test_pipeline_stages_3_4_6(self):
        """Test that helper functions work for stages 3, 4, and 6."""
        # Create test data
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:280, 250:390] = True

        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        labels = ["Team A Player", "#23 Smith", "#10 Johnson"]

        # Apply annotations for different stages
        results = []
        for color, label in zip(colors, labels):
            result = draw_player_annotation(
                frame.copy(), mask, color, label, alpha=TEAM_OVERLAY_ALPHA
            )
            results.append(result)

        # All results should be valid
        for result in results:
            assert result.shape == frame.shape
            assert result.dtype == np.uint8
            assert not np.array_equal(result, frame)

    def test_multiple_players_on_same_frame(self):
        """Test annotating multiple players on the same frame."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # Create two player masks
        mask1 = np.zeros((480, 640), dtype=bool)
        mask1[100:200, 100:200] = True

        mask2 = np.zeros((480, 640), dtype=bool)
        mask2[250:350, 400:500] = True

        # Draw both annotations
        annotated = frame.copy()
        annotated = draw_player_annotation(annotated, mask1, (0, 255, 0), "Player 1")
        annotated = draw_player_annotation(annotated, mask2, (0, 0, 255), "Player 2")

        # Frame should be significantly different
        assert not np.array_equal(annotated, frame)

        # Check that both masked regions have changed
        assert not np.array_equal(annotated[150, 150], frame[150, 150])
        assert not np.array_equal(annotated[300, 450], frame[300, 450])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
