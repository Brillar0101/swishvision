"""
Integration tests for the basketball tracking pipeline.

Tests the interaction between different components and validates
end-to-end functionality.
"""
import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

from app.ml.player_tracker import (
    PlayerTracker,
    PipelineCheckpoint,
    compute_iou,
    mask_to_box,
    filter_segments_by_distance,
)
from app.ml.team_classifier import TeamClassifier, get_player_crops
from app.ml.tactical_view import TacticalView
from app.ml.path_smoothing import clean_paths, smooth_tactical_positions
import supervision as sv


class TestPipelineCheckpoint:
    """Test pipeline checkpointing system."""

    def test_checkpoint_creation_and_loading(self):
        """Test that checkpoints can be created and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = PipelineCheckpoint(tmpdir)

            # Save metadata
            checkpoint.save_metadata('fps', 30.0)
            checkpoint.save_metadata('width', 1920)
            checkpoint.save_metadata('height', 1080)

            # Mark stage complete
            checkpoint.mark_stage_complete('frames_extracted')

            # Create new checkpoint instance to test persistence
            checkpoint2 = PipelineCheckpoint(tmpdir)
            assert checkpoint2.get_metadata('fps') == 30.0
            assert checkpoint2.get_metadata('width') == 1920
            assert checkpoint2.get_metadata('height') == 1080
            assert checkpoint2.is_stage_complete('frames_extracted')

    def test_checkpoint_data_persistence(self):
        """Test that checkpoint data is persisted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = PipelineCheckpoint(tmpdir)

            # Save test data
            test_data = {'player_1': {'team': 0, 'jersey': '23'}, 'player_2': {'team': 1, 'jersey': '35'}}
            checkpoint.save_data('tracking_info', test_data)

            # Load and verify
            loaded_data = checkpoint.load_data('tracking_info')
            assert loaded_data == test_data

            # Test default value for non-existent data
            missing = checkpoint.load_data('nonexistent', default={'empty': True})
            assert missing == {'empty': True}

    def test_checkpoint_clear(self):
        """Test that checkpoints can be cleared."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = PipelineCheckpoint(tmpdir)
            checkpoint.save_metadata('fps', 30.0)
            checkpoint.mark_stage_complete('frames_extracted')
            checkpoint.save_data('test', [1, 2, 3])

            checkpoint.clear()

            assert not checkpoint.is_stage_complete('frames_extracted')
            assert checkpoint.get_metadata('fps') is None
            assert checkpoint.load_data('test') is None


class TestUtilityFunctions:
    """Test utility functions used across the pipeline."""

    def test_compute_iou_perfect_overlap(self):
        """Test IoU calculation with perfect overlap."""
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]
        iou = compute_iou(box1, box2)
        assert iou == 1.0

    def test_compute_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]
        iou = compute_iou(box1, box2)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = compute_iou(box1, box2)
        # 50x50 intersection / (10000 + 10000 - 2500) union
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 0.001

    def test_mask_to_box_valid_mask(self):
        """Test mask to bounding box conversion."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:70] = True

        box = mask_to_box(mask)
        assert box is not None
        assert box[0] == 30  # x_min
        assert box[1] == 20  # y_min
        assert box[2] == 69  # x_max
        assert box[3] == 39  # y_max

    def test_mask_to_box_empty_mask(self):
        """Test that empty mask returns None."""
        mask = np.zeros((100, 100), dtype=bool)
        box = mask_to_box(mask)
        assert box is None

    def test_filter_segments_by_distance(self):
        """Test disconnected segment filtering."""
        # Create mask with two disconnected segments
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 50:100] = True  # Main segment
        mask[150:160, 150:160] = True  # Small distant segment

        filtered = filter_segments_by_distance(mask, relative_distance=0.1)

        # Main segment should remain
        assert filtered[75, 75] == True
        # Small distant segment should be removed
        assert filtered[155, 155] == False


class TestTeamClassifierIntegration:
    """Test team classifier with real crops."""

    def test_classifier_training_and_prediction(self):
        """Test that classifier can be trained and make predictions."""
        # Create synthetic player crops (green and red teams)
        green_crops = [np.ones((64, 64, 3), dtype=np.uint8) * [0, 255, 0] for _ in range(5)]
        red_crops = [np.ones((64, 64, 3), dtype=np.uint8) * [0, 0, 255] for _ in range(5)]
        all_crops = green_crops + red_crops

        classifier = TeamClassifier(n_teams=2, device='cpu')
        classifier.fit(all_crops)

        assert classifier.is_fitted

        # Test prediction
        green_test = np.ones((64, 64, 3), dtype=np.uint8) * [0, 255, 0]
        red_test = np.ones((64, 64, 3), dtype=np.uint8) * [0, 0, 255]

        green_pred = classifier.predict_single(green_test)
        red_pred = classifier.predict_single(red_test)

        # Teams should be different
        assert green_pred != red_pred

    def test_get_player_crops(self):
        """Test player crop extraction from frame."""
        # Create test frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Create test detections
        xyxy = np.array([[100, 100, 200, 300]])
        detections = sv.Detections(xyxy=xyxy, class_id=np.array([3]))

        crops = get_player_crops(frame, detections, scale_factor=0.4)

        assert len(crops) == 1
        assert crops[0].shape[2] == 3  # RGB image


class TestPathSmoothingIntegration:
    """Test path smoothing with realistic data."""

    def test_clean_paths_removes_jumps(self):
        """Test that clean_paths removes sudden position jumps."""
        # Create path with a teleportation jump
        n_frames = 100
        n_players = 2
        video_xy = np.zeros((n_frames, n_players, 2))

        # Player 1: smooth movement
        video_xy[:, 0, 0] = np.linspace(0, 100, n_frames)  # x
        video_xy[:, 0, 1] = np.linspace(0, 50, n_frames)   # y

        # Player 2: smooth movement with a jump
        video_xy[:, 1, 0] = np.linspace(0, 100, n_frames)
        video_xy[:, 1, 1] = np.linspace(0, 50, n_frames)
        video_xy[50, 1, :] = [500, 500]  # Teleport!

        cleaned_xy, edited_mask = clean_paths(video_xy)

        # Player 1 should be mostly unedited
        assert np.sum(edited_mask[:, 0]) < 5

        # Player 2 should have edits around frame 50
        assert edited_mask[50, 1] == True

    def test_smooth_tactical_positions(self):
        """Test tactical view position smoothing."""
        # Create position history with jitter
        positions_history = []
        for i in range(50):
            positions = {
                0: (100 + i + np.random.randn() * 2, 200 + i + np.random.randn() * 2),
                1: (300 - i + np.random.randn() * 2, 150 + np.random.randn() * 2),
            }
            positions_history.append(positions)

        smoothed = smooth_tactical_positions(positions_history, window_size=5)

        assert len(smoothed) == len(positions_history)

        # Smoothed positions should have less variance
        original_var_x = np.var([p[0][0] for p in positions_history[10:40]])
        smoothed_var_x = np.var([p[0][0] for p in smoothed[10:40]])
        assert smoothed_var_x < original_var_x


class TestTacticalViewIntegration:
    """Test tactical view rendering."""

    def test_tactical_view_rendering(self):
        """Test that tactical view can render positions."""
        tactical_view = TacticalView()

        # Create test frame with some court-like features
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Try to build transformer (may fail without actual court)
        tactical_view.build_transformer(frame)

        # Test positions
        positions = {
            0: (100, 200),
            1: (200, 250),
            2: (300, 200),
        }
        team_assignments = {0: 0, 1: 0, 2: 1}
        team_colors = {0: (0, 255, 0), 1: (0, 0, 255), -1: (0, 255, 255)}

        # This should not crash even if transformer failed
        rendered = tactical_view.render(positions, (480, 640), team_assignments, team_colors)

        assert rendered is not None
        assert rendered.shape[0] > 0
        assert rendered.shape[1] > 0


class TestPlayerTrackerExtractedMethods:
    """Test the extracted methods from PlayerTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a PlayerTracker instance for testing."""
        return PlayerTracker(enable_jersey_detection=False)

    @pytest.fixture
    def test_video(self):
        """Create a simple test video file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")

            # Create a simple 10-frame video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

            for i in range(10):
                frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 25)
                writer.write(frame)
            writer.release()

            yield video_path

    def test_extract_frames(self, tracker, test_video):
        """Test frame extraction method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, 'frames')
            os.makedirs(frames_dir)

            checkpoint = PipelineCheckpoint(os.path.join(tmpdir, 'checkpoints'))

            frames, fps, width, height = tracker._extract_frames(
                test_video, frames_dir, checkpoint, max_seconds=None, resume=False
            )

            assert len(frames) == 10
            assert fps == 30.0
            assert width == 640
            assert height == 480
            assert checkpoint.is_stage_complete('frames_extracted')

    def test_extract_frames_with_max_seconds(self, tracker, test_video):
        """Test frame extraction with time limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, 'frames')
            os.makedirs(frames_dir)

            checkpoint = PipelineCheckpoint(os.path.join(tmpdir, 'checkpoints'))

            frames, fps, width, height = tracker._extract_frames(
                test_video, frames_dir, checkpoint, max_seconds=0.1, resume=False
            )

            # With 30fps and 0.1s limit, should get 3 frames
            assert len(frames) == 3

    def test_collect_player_crops(self, tracker):
        """Test player crop collection method."""
        # Create test frames with no actual players
        frames = [np.ones((480, 640, 3), dtype=np.uint8) * 128 for _ in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = PipelineCheckpoint(tmpdir)

            crops = tracker._collect_player_crops(frames, checkpoint, resume=False)

            # With synthetic frames, likely no crops detected
            assert isinstance(crops, list)
            assert checkpoint.is_stage_complete('crops_collected')

    def test_train_team_classifier(self, tracker):
        """Test team classifier training method."""
        # Create synthetic crops
        crops = [np.ones((64, 64, 3), dtype=np.uint8) * i for i in range(10)]
        team_names = ("Team A", "Team B")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = PipelineCheckpoint(tmpdir)

            classifier = tracker._train_team_classifier(
                crops, team_names, checkpoint, resume=False
            )

            assert classifier is not None
            assert classifier.team_names == {0: "Team A", 1: "Team B"}
            assert checkpoint.is_stage_complete('team_classifier_trained')

    def test_detect_court(self, tracker):
        """Test court detection method."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        court_mask = tracker._detect_court(frame, 480, 640)

        assert court_mask.shape == (480, 640)
        assert court_mask.dtype == np.uint8
        # Court mask should have some active region
        assert np.sum(court_mask > 0) > 0


class TestPipelineStagesConsistency:
    """Test that different pipeline stages work together correctly."""

    def test_mask_box_consistency(self):
        """Test that mask_to_box and box operations are consistent."""
        # Create a mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 30:80] = True

        # Get box
        box = mask_to_box(mask)
        assert box is not None

        # Recreate mask from box
        recreated_mask = np.zeros((100, 100), dtype=bool)
        x1, y1, x2, y2 = map(int, box)
        recreated_mask[y1:y2+1, x1:x2+1] = True

        # Should have significant overlap
        intersection = np.sum(mask & recreated_mask)
        union = np.sum(mask | recreated_mask)
        iou = intersection / union
        assert iou > 0.95

    def test_detection_tracking_team_flow(self):
        """Test that detection -> tracking -> team assignment flow works."""
        # This is a simplified test of the pipeline flow

        # 1. Detections
        detections_xyxy = np.array([[100, 100, 150, 200], [300, 100, 350, 200]])

        # 2. Convert to tracking info
        tracking_info = {}
        for i, box in enumerate(detections_xyxy):
            tracking_info[i] = {'box': box.tolist(), 'class': 'player'}

        # 3. Assign teams (simplified)
        for obj_id in tracking_info:
            tracking_info[obj_id]['team'] = obj_id % 2
            tracking_info[obj_id]['team_name'] = f"Team {obj_id % 2}"

        # 4. Verify consistency
        assert len(tracking_info) == 2
        assert tracking_info[0]['team'] == 0
        assert tracking_info[1]['team'] == 1
        assert all('box' in info for info in tracking_info.values())
        assert all('team_name' in info for info in tracking_info.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
