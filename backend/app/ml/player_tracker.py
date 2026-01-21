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
import tempfile
import shutil

from sam2.build_sam import build_sam2_video_predictor

from app.ml.player_detector import PlayerDetector
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

        self.player_detector = PlayerDetector()
        self.court_detector = CourtDetector()

        # Jersey detection components
        self._jersey_detector = None
        self._jersey_ocr_model = None
        self.number_validator = ConsecutiveValueTracker(n_consecutive=3)
        self.team_validator = ConsecutiveValueTracker(n_consecutive=1)

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
        max_seconds: float = 10.0,
        team_names: Tuple[str, str] = ("Indiana Pacers", "Oklahoma City Thunder"),
        jersey_ocr_interval: int = 5,
        smooth_tactical: bool = True,
    ) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Reset validators
        self.number_validator.reset()
        self.team_validator.reset()

        try:
            print("Extracting frames...")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            max_frames = int(fps * max_seconds)
            if len(frames) > max_frames:
                frames = frames[:max_frames]

            frame_count = len(frames)
            print(f"Loaded {frame_count} frames ({frame_count/fps:.1f}s)")

            for idx, frame in enumerate(frames):
                cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.jpg"), frame)

            print("Collecting player crops...")
            all_crops = []
            for i in range(0, len(frames), 30):
                sv_detections = self.player_detector.detect(frames[i])
                players = sv_detections[np.isin(sv_detections.class_id, PLAYER_CLASS_IDS)]
                if len(players) > 0:
                    crops = get_player_crops(frames[i], players, scale_factor=0.4)
                    all_crops.extend(crops)
            print(f"  Collected {len(all_crops)} crops")

            print("Training team classifier...")
            team_classifier = TeamClassifier(n_teams=2, device="cpu")
            if len(all_crops) >= 2:
                team_classifier.fit(all_crops)

            # Set team names
            team_classifier.team_names = {0: team_names[0], 1: team_names[1]}

            print("Detecting court...")
            court_result = self.court_detector.detect_keypoints(frames[0])
            print(f"  Court keypoints: {court_result['count'] if court_result else 0}")

            court_mask = np.zeros((height, width), dtype=np.uint8)
            court_mask[int(height * 0.20):int(height * 0.85), :] = 255

            keyframe_indices = list(range(0, len(frames), keyframe_interval))
            print(f"Will detect at {len(keyframe_indices)} keyframes")

            print("Loading SAM2 model...")
            predictor = build_sam2_video_predictor(
                config_file=f"configs/sam2.1/{self.sam2_config}.yaml",
                ckpt_path=self.sam2_checkpoint,
                device=self.device,
            )

            print("Initializing video tracking...")
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.device.type=="cuda"):
                inference_state = predictor.init_state(video_path=frames_dir)

                tracking_info = {}
                current_boxes = {}
                next_obj_id = 0

                for kf_idx in keyframe_indices:
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

                    print(f"  Frame {kf_idx}: {len(filtered)} on-court, +{new_count} new (total: {next_obj_id})")

                print(f"Total tracking targets: {len(tracking_info)}")

                if len(tracking_info) == 0:
                    return {"error": "No players detected"}

                print("Propagating masks...")
                video_segments = {}

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
                    if out_frame_idx % 50 == 0:
                        print(f"    Frame {out_frame_idx}/{frame_count}")

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

            # Jersey number detection pass
            jersey_numbers = {}
            if self.enable_jersey_detection:
                print("Detecting jersey numbers...")
                for frame_idx in range(0, frame_count, jersey_ocr_interval):
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

                    if frame_idx % 50 == 0:
                        validated = self.number_validator.get_all_validated()
                        print(f"    Frame {frame_idx}: {len(validated)} numbers validated")

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
                tactical_view.build_transformer(frame)

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

            # First pass: collect all positions
            print("Collecting positions for smoothing...")
            for frame_idx in range(frame_count):
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

            sample_indices = [int(i * (frame_count - 1) / (num_sample_frames - 1)) for i in range(num_sample_frames)]

            print("Generating sample frames...")
            sample_frames = []
            for idx, frame_idx in enumerate(sample_indices):
                smoothed = smoothed_positions[frame_idx] if frame_idx < len(smoothed_positions) else None
                annotated, _ = annotate_frame(frames[frame_idx], frame_idx, smoothed)
                path = os.path.join(output_dir, f"tracking_frame_{idx+1:02d}.jpg")
                cv2.imwrite(path, annotated)
                sample_frames.append(path)
            print(f"  Saved {len(sample_frames)} frames")

            print("Generating video...")
            output_video_path = os.path.join(output_dir, "tracking_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            for frame_idx in range(frame_count):
                smoothed = smoothed_positions[frame_idx] if frame_idx < len(smoothed_positions) else None
                annotated, _ = annotate_frame(frames[frame_idx], frame_idx, smoothed)
                out_video.write(annotated)

            out_video.release()
            print(f"Video saved: {output_video_path}")

            return {
                "video_path": output_video_path,
                "sample_frames": sample_frames,
                "total_frames": frame_count,
                "players_tracked": len(tracking_info),
                "tracking_info": tracking_info,
                "jersey_numbers": jersey_numbers,
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
