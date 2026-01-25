"""
Player tracking using SAM2 for video segmentation.
Includes team classification, jersey number detection, and tactical view.

Based on Roboflow's basketball AI notebook implementation.
"""
import os
import cv2
import numpy as np
import torch
import supervision as sv
from typing import Dict, List, Tuple, Optional
import pickle
import json
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor

from app.ml.player_referee_detector import PlayerRefereeDetector
from app.ml.court_detector import CourtDetector
from app.ml.team_classifier import TeamClassifier, get_player_crops
from app.ml.tactical_view import TacticalView, create_combined_view
from app.ml.team_rosters import TEAM_ROSTERS, TEAM_COLORS, get_player_name
from app.ml.path_smoothing import smooth_tactical_positions

PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]
REFEREE_CLASS_IDS = [0, 1, 2]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def mask_to_box(mask):
    mask_2d = mask.squeeze()
    ys, xs = np.where(mask_2d)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def filter_segments_by_distance(mask: np.ndarray, relative_distance: float = 0.03) -> np.ndarray:
    """
    Filter out small disconnected segments from a mask.
    Keeps only the largest segment and removes smaller ones beyond a distance threshold.
    """
    from scipy import ndimage

    mask_2d = mask.squeeze().astype(np.uint8)
    labeled, num_features = ndimage.label(mask_2d)

    if num_features <= 1:
        return mask

    # Find the largest segment
    sizes = ndimage.sum(mask_2d, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1

    # Get centroid of largest segment
    largest_mask = labeled == largest_label
    ys, xs = np.where(largest_mask)
    if len(xs) == 0:
        return mask

    centroid = (np.mean(xs), np.mean(ys))
    max_dist = max(mask_2d.shape) * relative_distance

    # Keep segments close to the largest one
    result = np.zeros_like(mask_2d)
    for label_id in range(1, num_features + 1):
        segment_mask = labeled == label_id
        seg_ys, seg_xs = np.where(segment_mask)
        if len(seg_xs) == 0:
            continue

        seg_centroid = (np.mean(seg_xs), np.mean(seg_ys))
        dist = np.sqrt((seg_centroid[0] - centroid[0])**2 + (seg_centroid[1] - centroid[1])**2)

        if label_id == largest_label or dist <= max_dist:
            result[segment_mask] = 1

    return result.reshape(mask.shape).astype(bool)


class PipelineCheckpoint:
    """
    Manages checkpointing for the video processing pipeline.
    Saves intermediate results to disk so processing can resume after failures.
    """

    STAGES = [
        'frames_extracted',
        'crops_collected',
        'team_classifier_trained',
        'court_detected',
        'sam2_segmented',
        'teams_assigned',
        'jersey_detected',
        'positions_smoothed',
        'videos_generated',
    ]

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._status_file = os.path.join(checkpoint_dir, 'pipeline_status.json')
        self._load_status()

    def _load_status(self):
        """Load existing status or create new."""
        if os.path.exists(self._status_file):
            with open(self._status_file, 'r') as f:
                self._status = json.load(f)
        else:
            self._status = {'completed_stages': [], 'metadata': {}}

    def _save_status(self):
        """Save status to disk."""
        with open(self._status_file, 'w') as f:
            json.dump(self._status, f, indent=2)

    def is_stage_complete(self, stage: str) -> bool:
        """Check if a stage has been completed."""
        return stage in self._status['completed_stages']

    def mark_stage_complete(self, stage: str):
        """Mark a stage as completed."""
        if stage not in self._status['completed_stages']:
            self._status['completed_stages'].append(stage)
            self._save_status()
            print(f"  [Checkpoint] Stage '{stage}' saved")

    def save_metadata(self, key: str, value):
        """Save metadata (fps, dimensions, etc.)."""
        self._status['metadata'][key] = value
        self._save_status()

    def get_metadata(self, key: str, default=None):
        """Get saved metadata."""
        return self._status['metadata'].get(key, default)

    def save_data(self, name: str, data):
        """Save data to checkpoint file using pickle."""
        path = os.path.join(self.checkpoint_dir, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, name: str, default=None):
        """Load data from checkpoint file."""
        path = os.path.join(self.checkpoint_dir, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return default

    def save_frames_dir(self, frames_dir: str):
        """Save reference to frames directory."""
        self._status['metadata']['frames_dir'] = frames_dir
        self._save_status()

    def get_frames_dir(self) -> Optional[str]:
        """Get frames directory if it exists."""
        frames_dir = self._status['metadata'].get('frames_dir')
        if frames_dir and os.path.exists(frames_dir):
            return frames_dir
        return None

    def clear(self):
        """Clear all checkpoints to start fresh."""
        import glob
        for f in glob.glob(os.path.join(self.checkpoint_dir, '*.pkl')):
            os.remove(f)
        if os.path.exists(self._status_file):
            os.remove(self._status_file)
        self._status = {'completed_stages': [], 'metadata': {}}
        print("  [Checkpoint] Cleared all checkpoints")

    def get_progress_summary(self) -> str:
        """Get a summary of completed stages."""
        completed = self._status['completed_stages']
        if not completed:
            return "No stages completed"
        return f"Completed: {', '.join(completed)}"


class ConsecutiveValueTracker:
    """
    Tracks values over time and validates them based on consecutive occurrences.
    A value is only confirmed after appearing n_consecutive times in a row.
    """

    def __init__(self, n_consecutive: int = 3):
        self.n_consecutive = n_consecutive
        self._current_values: Dict[int, str] = {}
        self._consecutive_counts: Dict[int, int] = {}
        self._validated_values: Dict[int, str] = {}

    def update(self, tracker_ids: List[int], values: List) -> None:
        """Update tracker with new observations."""
        for tracker_id, value in zip(tracker_ids, values):
            if tracker_id in self._validated_values:
                continue

            if self._current_values.get(tracker_id) == value:
                self._consecutive_counts[tracker_id] = self._consecutive_counts.get(tracker_id, 0) + 1
            else:
                self._current_values[tracker_id] = value
                self._consecutive_counts[tracker_id] = 1

            if self._consecutive_counts[tracker_id] >= self.n_consecutive:
                self._validated_values[tracker_id] = value

    def get_validated(self, tracker_ids: List[int]) -> List:
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


class PlayerTracker:
    def __init__(
        self,
        sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
        sam2_config: str = "sam2.1_hiera_l",
        device: str = None,
        enable_jersey_detection: bool = True,
    ):
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.enable_jersey_detection = enable_jersey_detection

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.player_detector = PlayerRefereeDetector()
        self.court_detector = CourtDetector()

        # Jersey detection components
        self._jersey_detector = None
        self._jersey_ocr_model = None
        self.number_validator = ConsecutiveValueTracker(n_consecutive=3)
        self.team_validator = ConsecutiveValueTracker(n_consecutive=1)

    def _generate_portfolio_frames(
        self,
        frames: List[np.ndarray],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        tracking_info: Dict[int, Dict],
        team_classifier: 'TeamClassifier',
        sample_indices: List[int],
        output_dir: str,
        tactical_view: 'TacticalView',
        smoothed_positions: List[Dict],
    ) -> List[str]:
        """
        Generate portfolio frames showing each pipeline stage.

        Stages:
        1. Raw frame (original)
        2. Detection (bounding boxes only)
        3. Segmentation (masks only)
        4. Team classification (colored masks by team)
        5. Jersey detection (with numbers and names)
        6. Tactical view (2D court only, fullscreen)
        """
        from app.ml.tactical_view import draw_court

        portfolio_paths = []
        height, width = frames[0].shape[:2]

        for sample_idx, frame_idx in tqdm(enumerate(sample_indices), total=len(sample_indices), desc="Portfolio stages"):
            frame = frames[frame_idx]
            masks = video_segments.get(frame_idx, {})
            positions = smoothed_positions[frame_idx] if frame_idx < len(smoothed_positions) else {}

            # Stage 1: Raw frame
            path = os.path.join(output_dir, f"portfolio_{sample_idx+1:02d}_1_raw.jpg")
            stage1 = frame.copy()
            self._add_stage_label(stage1, "Stage 1: Raw Input Frame")
            cv2.imwrite(path, stage1)
            portfolio_paths.append(path)

            # Stage 2: Detection (boxes only)
            path = os.path.join(output_dir, f"portfolio_{sample_idx+1:02d}_2_detection.jpg")
            stage2 = frame.copy()
            for obj_id, mask in masks.items():
                box = mask_to_box(mask)
                if box:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(stage2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(stage2, f"ID:{obj_id}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self._add_stage_label(stage2, "Stage 2: Player Detection (RF-DETR)")
            cv2.imwrite(path, stage2)
            portfolio_paths.append(path)

            # Stage 3: Segmentation (masks with unique colors)
            path = os.path.join(output_dir, f"portfolio_{sample_idx+1:02d}_3_segmentation.jpg")
            stage3 = frame.copy()
            colors = self._generate_colors(len(masks))
            for i, (obj_id, mask) in enumerate(masks.items()):
                mask_2d = mask.squeeze()
                color = colors[i % len(colors)]
                mask_colored = np.zeros_like(stage3)
                mask_colored[mask_2d] = color
                stage3 = cv2.addWeighted(stage3, 1.0, mask_colored, 0.5, 0)
                # Draw contour
                contours, _ = cv2.findContours(mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(stage3, contours, -1, color, 2)
            self._add_stage_label(stage3, "Stage 3: Player Segmentation (SAM2)")
            cv2.imwrite(path, stage3)
            portfolio_paths.append(path)

            # Stage 4: Team classification (colored by team)
            path = os.path.join(output_dir, f"portfolio_{sample_idx+1:02d}_4_teams.jpg")
            stage4 = frame.copy()
            for obj_id, mask in masks.items():
                info = tracking_info.get(obj_id, {})
                team_id = info.get('team', 0)
                color = team_classifier.get_team_color(team_id)
                mask_2d = mask.squeeze()

                mask_colored = np.zeros_like(stage4)
                mask_colored[mask_2d] = color
                stage4 = cv2.addWeighted(stage4, 1.0, mask_colored, 0.4, 0)

                box = mask_to_box(mask)
                if box:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(stage4, (x1, y1), (x2, y2), color, 2)
                    team_name = info.get('team_name', 'Unknown')
                    cv2.putText(stage4, team_name, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            self._add_stage_label(stage4, "Stage 4: Team Classification (SigLIP + K-means)")
            cv2.imwrite(path, stage4)
            portfolio_paths.append(path)

            # Stage 5: Jersey detection (numbers and names)
            path = os.path.join(output_dir, f"portfolio_{sample_idx+1:02d}_5_jersey.jpg")
            stage5 = frame.copy()
            for obj_id, mask in masks.items():
                info = tracking_info.get(obj_id, {})
                team_id = info.get('team', 0)
                color = team_classifier.get_team_color(team_id)
                mask_2d = mask.squeeze()

                mask_colored = np.zeros_like(stage5)
                mask_colored[mask_2d] = color
                stage5 = cv2.addWeighted(stage5, 1.0, mask_colored, 0.4, 0)

                box = mask_to_box(mask)
                if box:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(stage5, (x1, y1), (x2, y2), color, 2)

                    jersey = info.get('jersey_number')
                    player_name = info.get('player_name')
                    if jersey and player_name:
                        label = f"#{jersey} {player_name}"
                    elif jersey:
                        label = f"#{jersey}"
                    else:
                        label = f"#{obj_id}"

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(stage5, (x1, y1-h-10), (x1+w+6, y1), color, -1)
                    cv2.putText(stage5, label, (x1+3, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            self._add_stage_label(stage5, "Stage 5: Jersey Detection (RF-DETR + SmolVLM2 OCR)")
            cv2.imwrite(path, stage5)
            portfolio_paths.append(path)

            # Stage 6: Tactical view (full court, not overlay)
            path = os.path.join(output_dir, f"portfolio_{sample_idx+1:02d}_6_tactical.jpg")
            if positions and tactical_view._last_transformer is not None:
                ta = {oid: tracking_info.get(oid, {}).get('team', 0) for oid in positions}
                tc = {0: team_classifier.get_team_color(0), 1: team_classifier.get_team_color(1), -1: (0, 255, 255)}
                stage6 = tactical_view.render(positions, (height, width), ta, tc)
                # Resize to match video dimensions
                stage6 = cv2.resize(stage6, (width, height))
            else:
                # Fallback: draw empty court
                stage6 = draw_court(width, height)
                cv2.putText(stage6, "Court keypoints not detected", (width//4, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self._add_stage_label(stage6, "Stage 6: Tactical 2D View (Homography Transform)")
            cv2.imwrite(path, stage6)
            portfolio_paths.append(path)

        return portfolio_paths

    def _generate_stage_videos(
        self,
        frames: List[np.ndarray],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        tracking_info: Dict[int, Dict],
        team_classifier: 'TeamClassifier',
        output_dir: str,
        tactical_view: 'TacticalView',
        smoothed_positions: List[Dict],
        fps: float,
        width: int,
        height: int,
    ) -> Dict[str, str]:
        """
        Generate video files for each pipeline stage.

        Stages:
        1. Raw video (original)
        2. Detection (bounding boxes only)
        3. Segmentation (masks only)
        4. Team classification (colored masks by team)
        5. Jersey detection (with numbers and names)
        6. Final with tactical view overlay
        """
        from app.ml.tactical_view import create_combined_view

        stage_videos = {}
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_count = len(frames)

        # Stage 1: Raw video
        print("  Stage 1: Raw video...")
        path = os.path.join(output_dir, "stage_1_raw.mp4")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for frame in tqdm(frames, desc="Raw", leave=False):
            out_frame = frame.copy()
            self._add_stage_label(out_frame, "Stage 1: Raw Input")
            writer.write(out_frame)
        writer.release()
        stage_videos['raw'] = path

        # Stage 2: Detection (boxes only)
        print("  Stage 2: Detection video...")
        path = os.path.join(output_dir, "stage_2_detection.mp4")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for frame_idx, frame in tqdm(enumerate(frames), total=frame_count, desc="Detection", leave=False):
            out_frame = frame.copy()
            masks = video_segments.get(frame_idx, {})
            for obj_id, mask in masks.items():
                box = mask_to_box(mask)
                if box:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(out_frame, f"ID:{obj_id}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self._add_stage_label(out_frame, "Stage 2: Player Detection (RF-DETR)")
            writer.write(out_frame)
        writer.release()
        stage_videos['detection'] = path

        # Stage 3: Segmentation (masks with unique colors)
        print("  Stage 3: Segmentation video...")
        path = os.path.join(output_dir, "stage_3_segmentation.mp4")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        all_obj_ids = sorted(set(oid for masks in video_segments.values() for oid in masks.keys()))
        colors = self._generate_colors(len(all_obj_ids))
        color_map = {oid: colors[i % len(colors)] for i, oid in enumerate(all_obj_ids)}
        for frame_idx, frame in tqdm(enumerate(frames), total=frame_count, desc="Segmentation", leave=False):
            out_frame = frame.copy()
            masks = video_segments.get(frame_idx, {})
            for obj_id, mask in masks.items():
                mask_2d = mask.squeeze()
                color = color_map.get(obj_id, (0, 255, 0))
                mask_colored = np.zeros_like(out_frame)
                mask_colored[mask_2d] = color
                out_frame = cv2.addWeighted(out_frame, 1.0, mask_colored, 0.5, 0)
                contours, _ = cv2.findContours(mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out_frame, contours, -1, color, 2)
            self._add_stage_label(out_frame, "Stage 3: Player Segmentation (SAM2)")
            writer.write(out_frame)
        writer.release()
        stage_videos['segmentation'] = path

        # Stage 4: Team classification (colored by team)
        print("  Stage 4: Team classification video...")
        path = os.path.join(output_dir, "stage_4_teams.mp4")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for frame_idx, frame in tqdm(enumerate(frames), total=frame_count, desc="Teams", leave=False):
            out_frame = frame.copy()
            masks = video_segments.get(frame_idx, {})
            for obj_id, mask in masks.items():
                info = tracking_info.get(obj_id, {})
                team_id = info.get('team', 0)
                color = team_classifier.get_team_color(team_id)
                mask_2d = mask.squeeze()

                mask_colored = np.zeros_like(out_frame)
                mask_colored[mask_2d] = color
                out_frame = cv2.addWeighted(out_frame, 1.0, mask_colored, 0.4, 0)

                box = mask_to_box(mask)
                if box:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                    team_name = info.get('team_name', 'Unknown')
                    cv2.putText(out_frame, team_name, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            self._add_stage_label(out_frame, "Stage 4: Team Classification (SigLIP + K-means)")
            writer.write(out_frame)
        writer.release()
        stage_videos['teams'] = path

        # Stage 5: Jersey detection (numbers and names)
        print("  Stage 5: Jersey detection video...")
        path = os.path.join(output_dir, "stage_5_jersey.mp4")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for frame_idx, frame in tqdm(enumerate(frames), total=frame_count, desc="Jersey", leave=False):
            out_frame = frame.copy()
            masks = video_segments.get(frame_idx, {})
            for obj_id, mask in masks.items():
                info = tracking_info.get(obj_id, {})
                team_id = info.get('team', 0)
                color = team_classifier.get_team_color(team_id)
                mask_2d = mask.squeeze()

                mask_colored = np.zeros_like(out_frame)
                mask_colored[mask_2d] = color
                out_frame = cv2.addWeighted(out_frame, 1.0, mask_colored, 0.4, 0)

                box = mask_to_box(mask)
                if box:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)

                    jersey = info.get('jersey_number')
                    player_name = info.get('player_name')
                    if jersey and player_name:
                        label = f"#{jersey} {player_name}"
                    elif jersey:
                        label = f"#{jersey}"
                    else:
                        label = f"#{obj_id}"

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(out_frame, (x1, y1-h-10), (x1+w+6, y1), color, -1)
                    cv2.putText(out_frame, label, (x1+3, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            self._add_stage_label(out_frame, "Stage 5: Jersey Detection (RF-DETR + SmolVLM2 OCR)")
            writer.write(out_frame)
        writer.release()
        stage_videos['jersey'] = path

        # Stage 6: Final with tactical view overlay
        print("  Stage 6: Final video with tactical view...")
        path = os.path.join(output_dir, "stage_6_final.mp4")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for frame_idx, frame in tqdm(enumerate(frames), total=frame_count, desc="Final", leave=False):
            out_frame = frame.copy()
            masks = video_segments.get(frame_idx, {})
            positions = smoothed_positions[frame_idx] if frame_idx < len(smoothed_positions) else {}

            for obj_id, mask in masks.items():
                info = tracking_info.get(obj_id, {})
                team_id = info.get('team', 0)
                color = team_classifier.get_team_color(team_id)
                mask_2d = mask.squeeze()

                mask_colored = np.zeros_like(out_frame)
                mask_colored[mask_2d] = color
                out_frame = cv2.addWeighted(out_frame, 1.0, mask_colored, 0.4, 0)

                box = mask_to_box(mask)
                if box:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)

                    jersey = info.get('jersey_number')
                    player_name = info.get('player_name')
                    if jersey and player_name:
                        label = f"#{jersey} {player_name}"
                    elif jersey:
                        label = f"#{jersey}"
                    else:
                        label = f"#{obj_id}"

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(out_frame, (x1, y1-h-10), (x1+w+6, y1), color, -1)
                    cv2.putText(out_frame, label, (x1+3, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Add tactical view overlay
            if positions and tactical_view._last_transformer is not None:
                ta = {oid: tracking_info.get(oid, {}).get('team', 0) for oid in positions}
                tc = {0: team_classifier.get_team_color(0), 1: team_classifier.get_team_color(1), -1: (0, 255, 255)}
                tactical = tactical_view.render(positions, (height, width), ta, tc)
                out_frame = create_combined_view(out_frame, tactical)

            self._add_stage_label(out_frame, "Stage 6: Final Output with Tactical View")
            writer.write(out_frame)
        writer.release()
        stage_videos['final'] = path

        return stage_videos

    def _add_stage_label(self, frame: np.ndarray, text: str):
        """Add stage label to top of frame."""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors."""
        colors = []
        for i in range(max(n, 1)):
            hue = int(180 * i / max(n, 1))
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr))
        return colors if colors else [(0, 255, 0)]

    def _init_jersey_detection(self):
        """Initialize jersey detection models (lazy loading)."""
        if self._jersey_detector is not None:
            return True

        if not self.enable_jersey_detection:
            return False

        try:
            from inference import get_model

            print("Loading jersey detection models...")
            self._jersey_detector = get_model(model_id="basketball-player-detection-3-ycjdo/4")
            self._jersey_ocr_model = get_model(model_id="basketball-jersey-numbers-ocr/3")
            print("Jersey detection models loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load jersey detection models: {e}")
            print("Jersey detection will be disabled")
            self.enable_jersey_detection = False
            return False

    def _detect_jersey_numbers(
        self,
        frame: np.ndarray,
        player_masks: Dict[int, np.ndarray],
        player_boxes: Dict[int, List[float]],
    ) -> Dict[int, str]:
        """
        Detect and recognize jersey numbers for players.

        Uses IoS (Intersection over Smaller Area) to match number
        detections to player masks.
        """
        if not self._init_jersey_detection():
            return {}

        frame_h, frame_w = frame.shape[:2]

        # Detect number bounding boxes
        result = self._jersey_detector.infer(
            frame,
            confidence=0.4,
            iou_threshold=0.9
        )[0]
        detections = sv.Detections.from_inference(result)
        number_detections = detections[detections.class_id == 2]  # NUMBER_CLASS_ID

        if len(number_detections) == 0:
            return {}

        # Convert to masks for IoS calculation
        number_masks = sv.xyxy_to_mask(
            boxes=number_detections.xyxy,
            resolution_wh=(frame_w, frame_h)
        )

        # Build player mask array
        player_ids = list(player_masks.keys())
        if not player_ids:
            return {}

        player_mask_array = np.array([
            player_masks[pid].squeeze() for pid in player_ids
        ])

        # Calculate IoS
        try:
            iou_matrix = sv.mask_iou_batch(
                masks_true=player_mask_array,
                masks_detection=number_masks,
                overlap_metric=sv.OverlapMetric.IOS
            )
        except Exception:
            return {}

        # Match and recognize
        matches = {}
        for player_idx, player_id in enumerate(player_ids):
            for number_idx in range(len(number_masks)):
                if iou_matrix[player_idx, number_idx] >= 0.9:
                    # Crop and recognize
                    box = number_detections.xyxy[number_idx]
                    padded = sv.pad_boxes(xyxy=np.array([box]), px=10, py=10)[0]
                    clipped = sv.clip_boxes(xyxy=np.array([padded]), resolution_wh=(frame_w, frame_h))[0]

                    try:
                        crop = sv.crop_image(frame, clipped)
                        crop_resized = sv.resize_image(crop, resolution_wh=(224, 224))
                        result = self._jersey_ocr_model.predict(crop_resized, "Read the number.")[0]

                        if result and result.strip().isdigit():
                            matches[player_id] = result.strip()
                    except Exception:
                        pass
                    break

        return matches

    def _match_detections_to_existing(self, new_detections, existing_boxes, iou_threshold=0.3):
        if not existing_boxes:
            return [], new_detections

        matched, unmatched = [], []
        existing_ids = list(existing_boxes.keys())
        existing_box_list = [existing_boxes[eid] for eid in existing_ids]

        for det in new_detections:
            det_box = det['box']
            best_iou, best_match_id = 0, None

            for eid, ebox in zip(existing_ids, existing_box_list):
                iou = compute_iou(det_box, ebox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = eid

            if best_iou >= iou_threshold:
                det['matched_id'] = best_match_id
                matched.append(det)
            else:
                unmatched.append(det)

        return matched, unmatched

    def process_video_with_tracking(
        self,
        video_path: str,
        output_dir: str,
        keyframe_interval: int = 30,
        iou_threshold: float = 0.3,
        max_total_objects: int = 15,
        num_sample_frames: int = 3,
        max_seconds: float = None,
        team_names: Tuple[str, str] = ("Indiana Pacers", "Oklahoma City Thunder"),
        jersey_ocr_interval: int = 5,
        smooth_tactical: bool = True,
        resume: bool = True,
        clear_checkpoints: bool = False,
        use_bytetrack: bool = False,
        use_sam2_segmentation: bool = False,
    ) -> Dict:
        """
        Process video with player tracking, team classification, and jersey detection.

        Args:
            video_path: Path to input video file
            output_dir: Directory for output files
            keyframe_interval: Frames between player detection keyframes (ignored if use_bytetrack=True)
            iou_threshold: IoU threshold for matching detections
            max_total_objects: Maximum players to track
            num_sample_frames: Number of sample frames to save
            max_seconds: Limit video to this many seconds (None for full video)
            team_names: Tuple of team names (team1, team2)
            jersey_ocr_interval: Frames between jersey OCR attempts
            smooth_tactical: Apply smoothing to tactical view positions
            resume: If True, resume from last checkpoint. If False, start fresh.
            clear_checkpoints: If True, clear all checkpoints before starting
            use_bytetrack: If True, use frame-by-frame ByteTrack for tracking
            use_sam2_segmentation: If True with use_bytetrack, add SAM2 segmentation to ByteTrack results

        Returns:
            Dict with results including paths to output videos and frames
        """
        os.makedirs(output_dir, exist_ok=True)

        # Initialize checkpoint system
        checkpoint_dir = os.path.join(output_dir, '.checkpoints')
        checkpoint = PipelineCheckpoint(checkpoint_dir)

        if clear_checkpoints:
            checkpoint.clear()

        if resume and checkpoint.get_progress_summary() != "No stages completed":
            print(f"[Resume Mode] {checkpoint.get_progress_summary()}")
        else:
            print("[Fresh Start] No checkpoints found or resume disabled")

        # Use persistent frames directory for checkpointing
        frames_dir = os.path.join(output_dir, '.frames_cache')
        os.makedirs(frames_dir, exist_ok=True)

        # Reset validators
        self.number_validator.reset()
        self.team_validator.reset()

        try:
            # ============ STAGE: Frame Extraction ============
            if resume and checkpoint.is_stage_complete('frames_extracted'):
                print("Extracting frames... [CACHED]")
                fps = checkpoint.get_metadata('fps')
                width = checkpoint.get_metadata('width')
                height = checkpoint.get_metadata('height')
                frame_count = checkpoint.get_metadata('frame_count')
                # Load frames from disk
                frames = []
                for idx in range(frame_count):
                    frame_path = os.path.join(frames_dir, f"{idx:05d}.jpg")
                    if os.path.exists(frame_path):
                        frames.append(cv2.imread(frame_path))
                print(f"  Loaded {len(frames)} frames from cache")
            else:
                print("Extracting frames...")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video file: {video_path}")

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30.0
                    print(f"  Warning: Could not read FPS, defaulting to {fps}")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()

                if max_seconds is not None:
                    max_frames = int(fps * max_seconds)
                    if len(frames) > max_frames:
                        frames = frames[:max_frames]

                frame_count = len(frames)
                print(f"Loaded {frame_count} frames ({frame_count/fps:.1f}s)")

                print("Saving frames to disk...")
                for idx, frame in tqdm(enumerate(frames), total=len(frames), desc="Saving frames"):
                    cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.jpg"), frame)

                # Save metadata
                checkpoint.save_metadata('fps', fps)
                checkpoint.save_metadata('width', width)
                checkpoint.save_metadata('height', height)
                checkpoint.save_metadata('frame_count', frame_count)
                checkpoint.mark_stage_complete('frames_extracted')

            # ============ STAGE: Crop Collection ============
            if resume and checkpoint.is_stage_complete('crops_collected'):
                print("Collecting player crops... [CACHED]")
                all_crops = checkpoint.load_data('crops', [])
                print(f"  Loaded {len(all_crops)} crops from cache")
            else:
                print("Collecting player crops...")
                all_crops = []
                crop_indices = list(range(0, len(frames), 30))
                for i in tqdm(crop_indices, desc="Collecting crops"):
                    sv_detections = self.player_detector.detect(frames[i])
                    players = sv_detections[np.isin(sv_detections.class_id, PLAYER_CLASS_IDS)]
                    if len(players) > 0:
                        crops = get_player_crops(frames[i], players, scale_factor=0.4)
                        all_crops.extend(crops)
                print(f"  Collected {len(all_crops)} crops")
                checkpoint.save_data('crops', all_crops)
                checkpoint.mark_stage_complete('crops_collected')

            # ============ STAGE: Team Classifier Training ============
            tc_device = "cuda" if self.device.type == "cuda" else "cpu"
            team_classifier = TeamClassifier(n_teams=2, device=tc_device)

            if resume and checkpoint.is_stage_complete('team_classifier_trained'):
                print("Training team classifier... [CACHED]")
                tc_data = checkpoint.load_data('team_classifier')
                if tc_data:
                    team_classifier._kmeans = tc_data.get('_kmeans')
                    team_classifier.is_fitted = tc_data.get('is_fitted', False)
                    print(f"  Loaded team classifier from cache (fitted={team_classifier.is_fitted})")
            else:
                print("Training team classifier...")
                if len(all_crops) >= 2:
                    team_classifier.fit(all_crops)
                checkpoint.save_data('team_classifier', {
                    '_kmeans': team_classifier._kmeans,
                    'is_fitted': team_classifier.is_fitted,
                })
                checkpoint.mark_stage_complete('team_classifier_trained')

            # Set team names
            team_classifier.team_names = {0: team_names[0], 1: team_names[1]}

            # ============ STAGE: Court Detection ============
            print("Detecting court...")
            court_result = self.court_detector.detect_keypoints(frames[0])
            print(f"  Court keypoints: {court_result['count'] if court_result else 0}")

            court_mask = np.zeros((height, width), dtype=np.uint8)
            court_mask[int(height * 0.20):int(height * 0.85), :] = 255

            keyframe_indices = list(range(0, len(frames), keyframe_interval))
            print(f"Will detect at {len(keyframe_indices)} keyframes")

            # ============ STAGE: Detection & Tracking ============
            if use_bytetrack:
                # ByteTrack mode: frame-by-frame detection with tracking
                stage_name = 'bytetrack_tracked'
            else:
                # SAM2 mode: keyframe detection with mask propagation
                stage_name = 'sam2_segmented'

            if resume and checkpoint.is_stage_complete(stage_name):
                mode_name = "ByteTrack tracking" if use_bytetrack else "SAM2 segmentation"
                print(f"{mode_name}... [CACHED]")
                video_segments = checkpoint.load_data('video_segments', {})
                tracking_info = checkpoint.load_data('tracking_info', {})
                current_boxes = checkpoint.load_data('current_boxes', {})
                print(f"  Loaded {len(video_segments)} frames with {len(tracking_info)} tracked objects from cache")
            elif use_bytetrack:
                # ============ ByteTrack Mode: Frame-by-frame detection ============
                print("Running ByteTrack frame-by-frame detection...")

                # Step 1: Get ByteTrack detections for all frames
                bytetrack_detections = {}  # frame_idx -> sv.Detections with tracker_id
                tracking_info = {}
                current_boxes = {}

                for frame_idx in tqdm(range(len(frames)), desc="ByteTrack detection"):
                    frame = frames[frame_idx]
                    detections = self.player_detector.detect_and_track(frame)

                    if len(detections) == 0:
                        bytetrack_detections[frame_idx] = sv.Detections.empty()
                        continue

                    # Filter by court mask
                    valid_mask = []
                    for i in range(len(detections)):
                        x1, y1, x2, y2 = detections.xyxy[i]
                        center_y = int((y1 + y2) / 2)
                        center_x = int((x1 + x2) / 2)
                        center_y = min(max(center_y, 0), height - 1)
                        center_x = min(max(center_x, 0), width - 1)
                        on_court = court_mask[center_y, center_x] > 0
                        valid_mask.append(on_court)

                    detections = detections[np.array(valid_mask)]
                    bytetrack_detections[frame_idx] = detections

                    # Build tracking info
                    for i in range(len(detections)):
                        if detections.tracker_id is None:
                            continue
                        tracker_id = int(detections.tracker_id[i])
                        box = detections.xyxy[i].tolist()
                        class_id = int(detections.class_id[i])
                        conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0

                        current_boxes[tracker_id] = box
                        if tracker_id not in tracking_info:
                            cls_name = 'player' if class_id in PLAYER_CLASS_IDS else 'referee'
                            tracking_info[tracker_id] = {'class': cls_name, 'confidence': conf}

                print(f"Total tracked objects: {len(tracking_info)}")

                if len(tracking_info) == 0:
                    return {"error": "No players detected"}

                # Step 2: Optionally add SAM2 segmentation
                if use_sam2_segmentation:
                    print("Adding SAM2 segmentation to ByteTrack results...")
                    print("Loading SAM2 model...")
                    predictor = build_sam2_video_predictor(
                        config_file=f"configs/sam2.1/{self.sam2_config}.yaml",
                        ckpt_path=self.sam2_checkpoint,
                        device=self.device,
                    )

                    use_autocast = self.device.type == "cuda"
                    autocast_dtype = torch.float16 if use_autocast else torch.float32

                    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                        inference_state = predictor.init_state(video_path=frames_dir)

                        # Add all tracked objects to SAM2 at their first appearance
                        tracker_first_frame = {}
                        for frame_idx, detections in bytetrack_detections.items():
                            if len(detections) == 0 or detections.tracker_id is None:
                                continue
                            for i in range(len(detections)):
                                tracker_id = int(detections.tracker_id[i])
                                if tracker_id not in tracker_first_frame:
                                    tracker_first_frame[tracker_id] = frame_idx
                                    box = detections.xyxy[i].tolist()
                                    box_np = np.array([[box[0], box[1], box[2], box[3]]], dtype=np.float32)
                                    predictor.add_new_points_or_box(
                                        inference_state=inference_state,
                                        frame_idx=frame_idx,
                                        obj_id=tracker_id,
                                        box=box_np,
                                    )

                        print(f"Propagating SAM2 masks for {len(tracker_first_frame)} objects...")
                        video_segments = {}
                        pbar = tqdm(total=frame_count, desc="SAM2 mask propagation")

                        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                            masks_dict = {}
                            for i in range(len(out_obj_ids)):
                                obj_id = int(out_obj_ids[i])
                                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                                mask = filter_segments_by_distance(mask, relative_distance=0.03)
                                masks_dict[obj_id] = mask
                                box = mask_to_box(mask)
                                if box is not None:
                                    current_boxes[obj_id] = box
                            video_segments[out_frame_idx] = masks_dict
                            pbar.update(1)
                        pbar.close()
                else:
                    # No SAM2: use bounding box masks
                    print("Using bounding box masks (no SAM2 segmentation)...")
                    video_segments = {}
                    for frame_idx, detections in tqdm(bytetrack_detections.items(), desc="Creating box masks"):
                        masks_dict = {}
                        for i in range(len(detections)):
                            if detections.tracker_id is None:
                                continue
                            tracker_id = int(detections.tracker_id[i])
                            box = detections.xyxy[i].tolist()
                            mask = np.zeros((height, width), dtype=bool)
                            x1, y1, x2, y2 = map(int, box)
                            mask[y1:y2, x1:x2] = True
                            masks_dict[tracker_id] = mask
                        video_segments[frame_idx] = masks_dict

                # Save results
                print("Saving results...")
                checkpoint.save_data('video_segments', video_segments)
                checkpoint.save_data('tracking_info', tracking_info)
                checkpoint.save_data('current_boxes', current_boxes)
                checkpoint.mark_stage_complete(stage_name)
            else:
                print("Loading SAM2 model...")
                predictor = build_sam2_video_predictor(
                    config_file=f"configs/sam2.1/{self.sam2_config}.yaml",
                    ckpt_path=self.sam2_checkpoint,
                    device=self.device,
                )

                print("Initializing video tracking...")
                # Only use autocast on CUDA, not CPU
                # Use float16 instead of bfloat16 for better compatibility with SAM2
                use_autocast = self.device.type == "cuda"
                autocast_dtype = torch.float16 if use_autocast else torch.float32
                with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                    inference_state = predictor.init_state(video_path=frames_dir)

                    tracking_info = {}
                    current_boxes = {}
                    next_obj_id = 0

                    print("Adding tracking targets at keyframes...")
                    for kf_idx in tqdm(keyframe_indices, desc="Keyframe detection"):
                        if next_obj_id >= max_total_objects:
                            break

                        frame = frames[kf_idx]
                        sv_detections = self.player_detector.detect(frame)

                        detections = []
                        for i in range(len(sv_detections)):
                            box = sv_detections.xyxy[i].tolist()
                            conf = float(sv_detections.confidence[i]) if sv_detections.confidence is not None else 1.0
                            cls_id = int(sv_detections.class_id[i]) if sv_detections.class_id is not None else 0
                            cls_name = 'player' if cls_id in PLAYER_CLASS_IDS else 'referee'
                            detections.append({'box': box, 'confidence': conf, 'class': cls_name})

                        filtered = [d for d in detections if court_mask[
                            min(max(int((d['box'][1] + d['box'][3]) / 2), 0), height - 1),
                            min(max(int((d['box'][0] + d['box'][2]) / 2), 0), width - 1)
                        ] > 0]

                        matched, unmatched = self._match_detections_to_existing(filtered, current_boxes, iou_threshold)

                        new_count = 0
                        for det in unmatched:
                            if next_obj_id >= max_total_objects:
                                break

                            box = det['box']
                            box_np = np.array([[box[0], box[1], box[2], box[3]]], dtype=np.float32)

                            predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=kf_idx,
                                obj_id=next_obj_id,
                                box=box_np,
                            )

                            tracking_info[next_obj_id] = {'class': det['class'], 'confidence': det['confidence']}
                            current_boxes[next_obj_id] = box
                            next_obj_id += 1
                            new_count += 1

                    print(f"Total tracking targets: {len(tracking_info)}")

                    if len(tracking_info) == 0:
                        return {"error": "No players detected"}

                    print("Propagating masks through video...")
                    video_segments = {}
                    pbar = tqdm(total=frame_count, desc="Mask propagation")

                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                        masks_dict = {}
                        for i in range(len(out_obj_ids)):
                            obj_id = int(out_obj_ids[i])
                            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                            # Clean up mask segments
                            mask = filter_segments_by_distance(mask, relative_distance=0.03)

                            masks_dict[obj_id] = mask
                            box = mask_to_box(mask)
                            if box is not None:
                                current_boxes[obj_id] = box
                        video_segments[out_frame_idx] = masks_dict
                        pbar.update(1)
                    pbar.close()

                # Save SAM2 results (this is the most expensive stage!)
                print("Saving SAM2 segmentation results...")
                checkpoint.save_data('video_segments', video_segments)
                checkpoint.save_data('tracking_info', tracking_info)
                checkpoint.save_data('current_boxes', current_boxes)
                checkpoint.mark_stage_complete('sam2_segmented')

            # ============ STAGE: Team Assignment ============
            if resume and checkpoint.is_stage_complete('teams_assigned'):
                print("Assigning teams... [CACHED]")
                tracking_info = checkpoint.load_data('tracking_info_with_teams', tracking_info)
                print(f"  Loaded team assignments from cache")
            else:
                print("Assigning teams...")
                if 0 in video_segments and team_classifier.is_fitted:
                    for obj_id, mask in video_segments[0].items():
                        if obj_id not in tracking_info:
                            continue
                        if tracking_info[obj_id]['class'] == 'referee':
                            tracking_info[obj_id]['team'] = -1
                            tracking_info[obj_id]['team_name'] = 'Referee'
                            continue
                        box = mask_to_box(mask)
                        if box is None:
                            continue
                        det = sv.Detections(xyxy=np.array([box]), class_id=np.array([3]))
                        crops = get_player_crops(frames[0], det, scale_factor=0.4)
                        if crops:
                            team_id = team_classifier.predict_single(crops[0])
                            tracking_info[obj_id]['team'] = team_id
                            tracking_info[obj_id]['team_name'] = team_classifier.get_team_name(team_id)

                for obj_id, info in tracking_info.items():
                    if 'team' not in info:
                        info['team'] = 0
                        info['team_name'] = team_names[0]

                checkpoint.save_data('tracking_info_with_teams', tracking_info)
                checkpoint.mark_stage_complete('teams_assigned')

            # ============ STAGE: Jersey Detection ============
            jersey_numbers = {}
            if resume and checkpoint.is_stage_complete('jersey_detected'):
                print("Detecting jersey numbers... [CACHED]")
                jersey_numbers = checkpoint.load_data('jersey_numbers', {})
                tracking_info = checkpoint.load_data('tracking_info_with_jerseys', tracking_info)
                print(f"  Loaded {len(jersey_numbers)} jersey numbers from cache")
            else:
                if self.enable_jersey_detection:
                    print("Detecting jersey numbers...")
                    ocr_frame_indices = list(range(0, frame_count, jersey_ocr_interval))
                    for frame_idx in tqdm(ocr_frame_indices, desc="Jersey OCR"):
                        if frame_idx not in video_segments:
                            continue

                        frame = frames[frame_idx]
                        player_masks = {}
                        player_boxes = {}

                        for obj_id, mask in video_segments[frame_idx].items():
                            if obj_id in tracking_info and tracking_info[obj_id]['class'] == 'player':
                                player_masks[obj_id] = mask
                                box = mask_to_box(mask)
                                if box:
                                    player_boxes[obj_id] = box

                        if player_masks:
                            matches = self._detect_jersey_numbers(frame, player_masks, player_boxes)
                            if matches:
                                self.number_validator.update(
                                    tracker_ids=list(matches.keys()),
                                    values=list(matches.values())
                                )

                    jersey_numbers = self.number_validator.get_all_validated()
                    print(f"  Validated jersey numbers: {jersey_numbers}")

                # Store jersey numbers in tracking info
                for obj_id, number in jersey_numbers.items():
                    if obj_id in tracking_info:
                        tracking_info[obj_id]['jersey_number'] = number
                        team_name = tracking_info[obj_id].get('team_name')
                        if team_name and team_name in TEAM_ROSTERS:
                            player_name = get_player_name(team_name, number)
                            if player_name:
                                tracking_info[obj_id]['player_name'] = player_name

                checkpoint.save_data('jersey_numbers', jersey_numbers)
                checkpoint.save_data('tracking_info_with_jerseys', tracking_info)
                checkpoint.mark_stage_complete('jersey_detected')

            print("Initializing tactical view...")
            tactical_view = TacticalView()

            # Collect positions for smoothing
            positions_history = []

            def get_player_label(obj_id: int, info: dict) -> str:
                """Generate label for a player."""
                jersey = info.get('jersey_number')
                player_name = info.get('player_name')
                team_name = info.get('team_name', '')

                if jersey and player_name:
                    return f"#{jersey} {player_name}"
                elif jersey:
                    return f"#{jersey}"
                else:
                    return f"#{obj_id} {team_name}"

            def annotate_frame(frame, frame_idx, smoothed_positions=None):
                annotated = frame.copy()
                player_positions = {}
                # NOTE: Don't call build_transformer here - it's already built once before this loop

                if frame_idx in video_segments:
                    for obj_id, mask in video_segments[frame_idx].items():
                        if obj_id not in tracking_info:
                            continue

                        info = tracking_info[obj_id]
                        mask_2d = mask.squeeze()
                        box = mask_to_box(mask)
                        if box is None:
                            continue

                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        player_positions[obj_id] = ((x1 + x2) / 2, y2)

                        team_id = info.get('team', 0)
                        color = team_classifier.get_team_color(team_id)

                        mask_colored = np.zeros_like(annotated)
                        mask_colored[mask_2d] = color
                        annotated = cv2.addWeighted(annotated, 1.0, mask_colored, 0.4, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                        label = get_player_label(obj_id, info)

                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Use smoothed positions if available
                render_positions = smoothed_positions if smoothed_positions else player_positions

                if render_positions:
                    ta = {oid: tracking_info[oid].get('team', 0) for oid in render_positions if oid in tracking_info}
                    tc = {0: team_classifier.get_team_color(0), 1: team_classifier.get_team_color(1), -1: (0, 255, 255)}
                    tactical = tactical_view.render(render_positions, (height, width), ta, tc)
                    annotated = create_combined_view(annotated, tactical)

                return annotated, player_positions

            # ============ STAGE: Position Smoothing ============
            if resume and checkpoint.is_stage_complete('positions_smoothed'):
                print("Collecting and smoothing positions... [CACHED]")
                smoothed_positions = checkpoint.load_data('smoothed_positions', [])
                print(f"  Loaded {len(smoothed_positions)} smoothed position frames from cache")
            else:
                # First pass: collect all positions
                print("Collecting positions for smoothing...")
                for frame_idx in tqdm(range(frame_count), desc="Collecting positions"):
                    if frame_idx in video_segments:
                        positions = {}
                        for obj_id, mask in video_segments[frame_idx].items():
                            box = mask_to_box(mask)
                            if box:
                                x1, y1, x2, y2 = box
                                positions[obj_id] = ((x1 + x2) / 2, y2)
                        positions_history.append(positions)
                    else:
                        positions_history.append({})

                # Smooth positions
                if smooth_tactical and len(positions_history) > 5:
                    print("Smoothing tactical positions...")
                    smoothed_positions = smooth_tactical_positions(positions_history, window_size=5)
                else:
                    smoothed_positions = positions_history

                checkpoint.save_data('smoothed_positions', smoothed_positions)
                checkpoint.mark_stage_complete('positions_smoothed')

            sample_indices = [int(i * (frame_count - 1) / (num_sample_frames - 1)) for i in range(num_sample_frames)]

            # Build transformer on first frame for tactical view
            print("Building court transformer...")
            tactical_view.build_transformer(frames[0])
            if tactical_view._last_transformer is not None:
                print("  Court transformer built successfully")
            else:
                print("  WARNING: Court transformer failed - tactical view will be empty")

            print("Generating sample frames...")
            sample_frames = []
            for idx, frame_idx in tqdm(enumerate(sample_indices), total=len(sample_indices), desc="Sample frames"):
                smoothed = smoothed_positions[frame_idx] if frame_idx < len(smoothed_positions) else None
                annotated, _ = annotate_frame(frames[frame_idx], frame_idx, smoothed)
                path = os.path.join(output_dir, f"tracking_frame_{idx+1:02d}.jpg")
                cv2.imwrite(path, annotated)
                sample_frames.append(path)
            print(f"  Saved {len(sample_frames)} sample frames")

            # Generate portfolio stage frames
            print("Generating portfolio stage frames...")
            portfolio_frames = self._generate_portfolio_frames(
                frames, video_segments, tracking_info, team_classifier,
                sample_indices, output_dir, tactical_view, smoothed_positions
            )
            print(f"  Saved {len(portfolio_frames)} portfolio frames")

            # Generate video outputs for each pipeline stage
            print("Generating stage videos...")
            stage_videos = self._generate_stage_videos(
                frames, video_segments, tracking_info, team_classifier,
                output_dir, tactical_view, smoothed_positions, fps, width, height
            )
            print(f"  Generated {len(stage_videos)} stage videos")
            output_video_path = stage_videos.get('final')

            return {
                "video_path": output_video_path,
                "stage_videos": stage_videos,
                "sample_frames": sample_frames,
                "portfolio_frames": portfolio_frames,
                "total_frames": frame_count,
                "players_tracked": len(tracking_info),
                "tracking_info": tracking_info,
                "jersey_numbers": jersey_numbers,
            }

        except Exception as e:
            print(f"\n[Pipeline Error] {e}")
            print(f"[Checkpoint] Progress saved. Resume with resume=True to continue from: {checkpoint.get_progress_summary()}")
            raise
