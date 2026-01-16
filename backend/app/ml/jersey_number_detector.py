"""
Jersey Number Detector - Pose-Guided Torso Cropping Approach
Uses:
- RF-DETR for player detection
- YOLO11 Pose for keypoint detection (shoulders, hips)
- Torso cropping from pose keypoints for precise OCR region
- PARSeq for state-of-the-art scene text recognition (jersey numbers)
- ConsecutiveValueTracker for 3-consecutive validation
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv
import supervision as sv
from inference import get_model

# PARSeq imports
try:
    from strhub.data.module import SceneTextDataModule
    from strhub.models.utils import load_from_checkpoint, parse_model_args
    HAS_PARSEQ = True
except ImportError:
    HAS_PARSEQ = False
    print("Warning: PARSeq (strhub) not installed. Install with: pip install strhub")

load_dotenv()

# Import ConsecutiveValueTracker from sports library
try:
    from sports import ConsecutiveValueTracker
    HAS_SPORTS_TRACKER = True
except ImportError:
    HAS_SPORTS_TRACKER = False
    print("Warning: sports.ConsecutiveValueTracker not available, using built-in")


# COCO Keypoint indices for torso
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12


class SimpleConsecutiveTracker:
    """Fallback if sports library not available"""
    def __init__(self, n_consecutive: int = 3):
        self.n_consecutive = n_consecutive
        self.history: Dict[int, List[str]] = defaultdict(list)
        self.validated: Dict[int, str] = {}

    def update(self, tracker_ids: List[int], values: List[str]) -> None:
        for tid, val in zip(tracker_ids, values):
            tid = int(tid)
            if tid in self.validated:
                continue  # Already validated, skip

            self.history[tid].append(val)

            # Keep only last n_consecutive
            if len(self.history[tid]) > self.n_consecutive:
                self.history[tid] = self.history[tid][-self.n_consecutive:]

            # Check if all last n are the same
            if len(self.history[tid]) >= self.n_consecutive:
                if len(set(self.history[tid][-self.n_consecutive:])) == 1:
                    self.validated[tid] = val
                    print(f"      ✓ Player #{val} confirmed (3 consecutive) - tracker {tid}")

    def get_validated(self, tracker_ids: List[int]) -> List[Optional[str]]:
        return [self.validated.get(int(tid), None) for tid in tracker_ids]

    def get_all_validated(self) -> Dict[int, str]:
        return self.validated.copy()


def extract_torso_bbox_from_keypoints(
    keypoints: np.ndarray,
    confidence_threshold: float = 0.3,
    padding_ratio: float = 0.15
) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract torso bounding box from pose keypoints.

    Args:
        keypoints: Array of shape (17, 3) with [x, y, confidence] for each keypoint
        confidence_threshold: Minimum confidence for keypoint to be considered valid
        padding_ratio: Extra padding around torso (0.15 = 15% padding)

    Returns:
        Tuple (x1, y1, x2, y2) or None if not enough keypoints detected
    """
    # Get torso keypoints: left/right shoulder and left/right hip
    left_shoulder = keypoints[KEYPOINT_LEFT_SHOULDER]
    right_shoulder = keypoints[KEYPOINT_RIGHT_SHOULDER]
    left_hip = keypoints[KEYPOINT_LEFT_HIP]
    right_hip = keypoints[KEYPOINT_RIGHT_HIP]

    # Check confidence for each keypoint
    valid_keypoints = []
    for kp in [left_shoulder, right_shoulder, left_hip, right_hip]:
        if len(kp) >= 3 and kp[2] >= confidence_threshold:
            valid_keypoints.append((kp[0], kp[1]))

    # Need at least 3 keypoints to estimate torso
    if len(valid_keypoints) < 3:
        return None

    # Get bounding box from valid keypoints
    xs = [p[0] for p in valid_keypoints]
    ys = [p[1] for p in valid_keypoints]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Add padding
    width = x2 - x1
    height = y2 - y1

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    x1 = int(x1 - pad_x)
    y1 = int(y1 - pad_y)
    x2 = int(x2 + pad_x)
    y2 = int(y2 + pad_y)

    return (x1, y1, x2, y2)


def extract_torso_crop(
    frame: np.ndarray,
    player_bbox: np.ndarray,
    keypoints: Optional[np.ndarray] = None,
    fallback_ratio: Tuple[float, float] = (0.2, 0.6)
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract torso crop from frame, using pose keypoints if available,
    otherwise falling back to estimated torso region from player bbox.

    Args:
        frame: Full video frame
        player_bbox: Player bounding box [x1, y1, x2, y2]
        keypoints: Optional pose keypoints array
        fallback_ratio: (top_ratio, bottom_ratio) for fallback crop

    Returns:
        Tuple of (cropped_image, absolute_bbox)
    """
    frame_h, frame_w = frame.shape[:2]
    px1, py1, px2, py2 = [int(v) for v in player_bbox]

    # Try to use keypoints first
    if keypoints is not None:
        torso_bbox = extract_torso_bbox_from_keypoints(keypoints)
        if torso_bbox is not None:
            x1, y1, x2, y2 = torso_bbox
            # Clip to frame bounds
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                return crop, (x1, y1, x2, y2)

    # Fallback: estimate torso from player bbox
    # Jersey numbers typically in top 20% to 60% of player height
    player_height = py2 - py1
    top_ratio, bottom_ratio = fallback_ratio

    y1 = py1 + int(player_height * top_ratio)
    y2 = py1 + int(player_height * bottom_ratio)
    x1, x2 = px1, px2

    # Clip to frame bounds
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w))
    y2 = max(0, min(y2, frame_h))

    if x2 <= x1 or y2 <= y1:
        # If invalid, just use whole player bbox
        x1, y1, x2, y2 = px1, py1, px2, py2
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))

    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


class PoseGuidedJerseyDetector:
    """
    State-of-the-art jersey number detection using pose-guided torso cropping.

    Pipeline:
    1. Detect players with RF-DETR
    2. Run YOLO11 Pose on each player crop to get keypoints
    3. Extract torso region using shoulder/hip keypoints
    4. Run OCR on precise torso crop
    5. Validate with 3-consecutive frame confirmation
    """

    def __init__(self, device: str = None, pose_model: str = "yolov8x-pose-640", use_parseq: bool = True):
        # Device setup
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"PoseGuidedJerseyDetector using device: {self.device}")

        # Load models
        print("Loading player detection model (RF-DETR)...")
        self.player_model = get_model(
            model_id="basketball-player-detection-3-ycjdo/6",
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )

        print(f"Loading pose estimation model ({pose_model})...")
        self.pose_model = get_model(
            model_id=pose_model,
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )

        # OCR model selection
        self.use_parseq = use_parseq and HAS_PARSEQ

        if self.use_parseq:
            print("Loading PARSeq model for jersey OCR...")
            self._init_parseq()
        else:
            print("Loading jersey OCR model (SmolVLM2 fallback)...")
            self.ocr_model = get_model(
                model_id="basketball-jersey-numbers-ocr/7",
                api_key=os.getenv("ROBOFLOW_API_KEY")
            )
            self.parseq_model = None
            self.parseq_transform = None

        # Class IDs for players
        self.PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]

        # Detection settings
        self.confidence = 0.4
        self.iou_threshold = 0.5
        self.pose_confidence = 0.3

        # Number validator (3 consecutive same reads to confirm)
        if HAS_SPORTS_TRACKER:
            self.number_validator = ConsecutiveValueTracker(n_consecutive=3)
        else:
            self.number_validator = SimpleConsecutiveTracker(n_consecutive=3)

        # Store confirmed numbers
        self.confirmed_numbers: Dict[int, str] = {}

        # Stats
        self.pose_success_count = 0
        self.fallback_count = 0

    def _init_parseq(self):
        """Initialize PARSeq model for scene text recognition."""
        try:
            import urllib.request
            import tempfile

            # Download PARSeq checkpoint directly
            weights_url = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt"
            weights_path = os.path.join(tempfile.gettempdir(), "parseq-bb5792a6.pt")

            if not os.path.exists(weights_path):
                print(f"    Downloading PARSeq weights...")
                urllib.request.urlretrieve(weights_url, weights_path)
                print(f"    Downloaded to {weights_path}")

            # Load state dict
            state_dict = torch.load(weights_path, map_location='cpu')

            from strhub.models.parseq.model import PARSeq as PARSeqModel
            from strhub.data.utils import Tokenizer

            # Standard charset (94 printable ASCII chars)
            # The pretrained model uses: digits + lowercase + uppercase + punctuation
            charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

            # Create tokenizer - adds 3 special tokens: [B], [E], [P]
            tokenizer = Tokenizer(charset)

            # Get actual num_tokens from weights (text_embed has vocab size)
            num_tokens = state_dict['text_embed.embedding.weight'].shape[0]

            self.parseq_model = PARSeqModel(
                num_tokens=num_tokens,
                max_label_length=25,
                img_size=(32, 128),
                patch_size=(4, 8),
                embed_dim=384,
                enc_num_heads=6,
                enc_mlp_ratio=4,
                enc_depth=12,
                dec_num_heads=12,
                dec_mlp_ratio=4,
                dec_depth=1,
                decode_ar=True,
                refine_iters=1,
                dropout=0.1,
            )

            # Load weights
            self.parseq_model.load_state_dict(state_dict, strict=True)
            self.parseq_model = self.parseq_model.eval().to(self.device)

            # Store tokenizer for decoding
            self.parseq_tokenizer = tokenizer

            # Image transform: resize to (32, 128) and normalize
            self.parseq_transform = SceneTextDataModule.get_transform((32, 128))

            print(f"    PARSeq loaded successfully on {self.device}")
        except Exception as e:
            print(f"    Failed to load PARSeq: {e}")
            import traceback
            traceback.print_exc()
            print("    Falling back to SmolVLM2...")
            self.use_parseq = False
            self.parseq_model = None
            self.parseq_transform = None
            self.parseq_tokenizer = None
            self.ocr_model = get_model(
                model_id="basketball-jersey-numbers-ocr/7",
                api_key=os.getenv("ROBOFLOW_API_KEY")
            )

    def get_pose_keypoints(self, player_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Run pose estimation on a player crop to get keypoints.

        Returns:
            Array of shape (17, 3) with [x, y, confidence] for each keypoint,
            or None if no person detected
        """
        try:
            result = self.pose_model.infer(player_crop)[0]

            # Parse keypoints from result
            if hasattr(result, 'predictions') and len(result.predictions) > 0:
                pred = result.predictions[0]
                if hasattr(pred, 'keypoints'):
                    keypoints = []
                    for kp in pred.keypoints:
                        keypoints.append([kp.x, kp.y, kp.confidence])
                    return np.array(keypoints)

            return None
        except Exception as e:
            print(f"        Pose estimation error: {e}")
            return None

    def run_ocr(self, crop: np.ndarray) -> Optional[str]:
        """Run OCR on a crop and return the detected number."""
        try:
            # Ensure crop is valid
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                return None

            if self.use_parseq and self.parseq_model is not None:
                return self._run_parseq_ocr(crop)
            else:
                return self._run_smolvlm_ocr(crop)

        except Exception as e:
            print(f"        OCR error: {e}")
            return None

    def _run_parseq_ocr(self, crop: np.ndarray) -> Optional[str]:
        """Run PARSeq OCR on a crop."""
        try:
            # Convert BGR (OpenCV) to RGB PIL Image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            # Apply transform and add batch dimension
            img_tensor = self.parseq_transform(pil_img).unsqueeze(0).to(self.device)

            # Run inference - PARSeq.forward needs tokenizer and images
            with torch.no_grad():
                logits = self.parseq_model(self.parseq_tokenizer, img_tensor)
                # logits shape: (batch, seq_len, num_classes)
                pred = logits.softmax(-1)

                # Decode using tokenizer - returns (labels, confidences)
                # confidences is a list of tensors (per-character confidence)
                labels, confidences = self.parseq_tokenizer.decode(pred)

            result_str = labels[0] if labels else ""

            # Extract only digits (jersey numbers are 1-2 digits)
            digits = ''.join(c for c in result_str if c.isdigit())

            # Validate: jersey numbers should be 1-2 digits
            if digits and len(digits) <= 2:
                # Get mean confidence across all characters
                conf = confidences[0].mean().item() if len(confidences) > 0 and confidences[0].numel() > 0 else 0
                if conf > 0.3:  # Accept moderate-confidence predictions
                    return digits

            return None
        except Exception as e:
            print(f"        PARSeq error: {e}")
            return None

    def _run_smolvlm_ocr(self, crop: np.ndarray) -> Optional[str]:
        """Run SmolVLM2 OCR on a crop (fallback)."""
        try:
            result = self.ocr_model.predict(
                crop,
                "Read the jersey number. If unclear or no number visible, respond with 'none'."
            )[0]

            # Clean result
            result_str = str(result).strip().lower()

            # Filter out invalid responses
            if result_str in ['none', 'n/a', 'unclear', '', 'null', '1']:
                return None

            # Try to extract just digits
            digits = ''.join(c for c in result_str if c.isdigit())
            if digits and len(digits) <= 2:
                return digits

            return None
        except Exception as e:
            print(f"        SmolVLM2 error: {e}")
            return None

    def detect_jersey_numbers(
        self,
        frame: np.ndarray,
        player_detections: sv.Detections,
    ) -> Dict[int, str]:
        """
        Detect jersey numbers using pose-guided torso cropping.

        Args:
            frame: Video frame
            player_detections: Player detections with tracker_id

        Returns:
            Dict mapping tracker_id -> jersey number string
        """
        import time

        if len(player_detections) == 0:
            return {}

        frame_h, frame_w = frame.shape[:2]

        detected_this_frame = []

        for i, (bbox, tracker_id) in enumerate(zip(
            player_detections.xyxy,
            player_detections.tracker_id
        )):
            tracker_id = int(tracker_id)

            # Skip if already confirmed
            if tracker_id in self.confirmed_numbers:
                continue

            # Extract player crop for pose estimation
            px1, py1, px2, py2 = [int(v) for v in bbox]
            px1 = max(0, px1)
            py1 = max(0, py1)
            px2 = min(frame_w, px2)
            py2 = min(frame_h, py2)

            if px2 <= px1 or py2 <= py1:
                continue

            player_crop = frame[py1:py2, px1:px2]

            # Run pose estimation on player crop
            t0 = time.time()
            keypoints = self.get_pose_keypoints(player_crop)

            # Get torso crop
            if keypoints is not None:
                # Adjust keypoints to absolute frame coordinates
                keypoints_abs = keypoints.copy()
                keypoints_abs[:, 0] += px1
                keypoints_abs[:, 1] += py1

                torso_crop, torso_bbox = extract_torso_crop(
                    frame, bbox, keypoints_abs
                )
                self.pose_success_count += 1
                crop_method = "pose"
            else:
                # Fallback to estimated torso region
                torso_crop, torso_bbox = extract_torso_crop(
                    frame, bbox, None
                )
                self.fallback_count += 1
                crop_method = "fallback"

            # Run OCR on torso crop
            jersey_num = self.run_ocr(torso_crop)

            if jersey_num:
                detected_this_frame.append((tracker_id, jersey_num))
                print(f"        Player {tracker_id}: #{jersey_num} ({crop_method})")

        # Update validator with this frame's detections
        if detected_this_frame:
            tracker_ids = [d[0] for d in detected_this_frame]
            numbers = [d[1] for d in detected_this_frame]
            self.number_validator.update(tracker_ids=tracker_ids, values=numbers)

        # Update confirmed numbers
        if HAS_SPORTS_TRACKER:
            validated = self.number_validator.get_validated(
                tracker_ids=list(player_detections.tracker_id)
            )
            for tid, num in zip(player_detections.tracker_id, validated):
                if num is not None:
                    self.confirmed_numbers[int(tid)] = num
        else:
            self.confirmed_numbers = self.number_validator.get_all_validated()

        return self.confirmed_numbers.copy()

    def process_frame(
        self,
        frame: np.ndarray,
        player_boxes: np.ndarray,
        tracker_ids: np.ndarray,
        player_masks: np.ndarray = None,
    ) -> Dict[int, str]:
        """
        Simplified interface for compatibility with player_tracker.py.
        """
        detections = sv.Detections(
            xyxy=np.array(player_boxes),
            tracker_id=np.array(tracker_ids).astype(int),
            mask=player_masks
        )

        return self.detect_jersey_numbers(frame, detections)

    def get_confirmed_numbers(self) -> Dict[int, str]:
        return self.confirmed_numbers.copy()

    def get_confirmed_count(self) -> int:
        return len(set(self.confirmed_numbers.values()))

    def get_stats(self) -> Dict:
        return {
            "pose_success": self.pose_success_count,
            "fallback": self.fallback_count,
            "confirmed_players": len(self.confirmed_numbers),
            "unique_jerseys": len(set(self.confirmed_numbers.values())),
            "ocr_model": "PARSeq" if self.use_parseq else "SmolVLM2"
        }

    def clear_cache(self):
        self.confirmed_numbers.clear()
        self.pose_success_count = 0
        self.fallback_count = 0
        if HAS_SPORTS_TRACKER:
            self.number_validator = ConsecutiveValueTracker(n_consecutive=3)
        else:
            self.number_validator = SimpleConsecutiveTracker(n_consecutive=3)


# Backwards compatibility alias
JerseyNumberDetector = PoseGuidedJerseyDetector


def process_video_jersey_numbers(
    video_path: str,
    output_dir: str,
    max_seconds: float = 99999.0,
    ocr_interval: int = 3,
    pose_model: str = "yolov8x-pose-640"
) -> Dict:
    """
    Process video with pose-guided jersey number detection.

    Args:
        video_path: Path to input video
        output_dir: Directory for outputs
        max_seconds: Maximum seconds to process
        ocr_interval: Run OCR every N frames
        pose_model: Pose estimation model to use
    """
    print("="*60)
    print("Jersey Number Detection - Pose-Guided Torso Cropping")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Load video
    print("\n[1] Loading video...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)

    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"    Loaded {len(frames)} frames ({len(frames)/fps:.1f}s)")

    # Initialize detector
    print("\n[2] Initializing pose-guided jersey detector...")
    jersey_detector = PoseGuidedJerseyDetector(pose_model=pose_model)

    # Simple IoU-based tracker
    print("\n[3] Processing frames...")

    PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]
    tracker_id_counter = 1
    prev_detections = None
    all_tracked = {}

    for frame_idx, frame in enumerate(frames):
        # Detect players
        result = jersey_detector.player_model.infer(
            frame,
            confidence=jersey_detector.confidence,
            iou_threshold=jersey_detector.iou_threshold
        )[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]

        # Assign tracker IDs
        if frame_idx == 0:
            detections.tracker_id = np.arange(1, len(detections) + 1)
            tracker_id_counter = len(detections) + 1
        else:
            if prev_detections is not None and len(prev_detections) > 0 and len(detections) > 0:
                iou_matrix = sv.box_iou_batch(detections.xyxy, prev_detections.xyxy)
                new_tracker_ids = []
                used_prev = set()

                for i in range(len(detections)):
                    best_iou = 0
                    best_prev_idx = -1
                    for j in range(len(prev_detections)):
                        if j not in used_prev and iou_matrix[i, j] > best_iou:
                            best_iou = iou_matrix[i, j]
                            best_prev_idx = j

                    if best_iou > 0.3 and best_prev_idx >= 0:
                        new_tracker_ids.append(prev_detections.tracker_id[best_prev_idx])
                        used_prev.add(best_prev_idx)
                    else:
                        new_tracker_ids.append(tracker_id_counter)
                        tracker_id_counter += 1

                detections.tracker_id = np.array(new_tracker_ids)
            else:
                detections.tracker_id = np.arange(tracker_id_counter, tracker_id_counter + len(detections))
                tracker_id_counter += len(detections)

        prev_detections = detections

        # Store tracking info
        all_tracked[frame_idx] = {
            int(tid): detections.xyxy[i].tolist()
            for i, tid in enumerate(detections.tracker_id)
        }

        # Run OCR every N frames
        if frame_idx % ocr_interval == 0 and len(detections) > 0:
            print(f"    Frame {frame_idx}: Running pose-guided OCR...")
            jersey_detector.detect_jersey_numbers(frame, detections)

            stats = jersey_detector.get_stats()
            print(f"    Frame {frame_idx}: {stats['confirmed_players']} players confirmed, "
                  f"{stats['pose_success']} pose / {stats['fallback']} fallback crops")

        # Progress every 30 frames
        if frame_idx % 30 == 0:
            print(f"    Processed {frame_idx}/{len(frames)} frames")

    # Final results
    confirmed_numbers = jersey_detector.get_confirmed_numbers()
    stats = jersey_detector.get_stats()

    print(f"\n[4] Results:")
    print(f"    OCR Model: {stats['ocr_model']}")
    print(f"    Unique players confirmed: {stats['unique_jerseys']}")
    print(f"    Jersey numbers: {list(set(confirmed_numbers.values()))}")
    print(f"    Pose-guided crops: {stats['pose_success']}")
    print(f"    Fallback crops: {stats['fallback']}")

    # Generate annotated video
    print("\n[5] Generating annotated video...")
    output_video_path = os.path.join(output_dir, "jersey_detection_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for frame_idx, frame in enumerate(frames):
        annotated = frame.copy()

        if frame_idx in all_tracked:
            for tid, box in all_tracked[frame_idx].items():
                jersey_num = confirmed_numbers.get(tid, None)
                if jersey_num is None:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]
                color = colors[tid % len(colors)]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"#{jersey_num}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_video.write(annotated)

    out_video.release()
    print(f"    Video saved: {output_video_path}")

    # Save tracking data
    print("\n[6] Saving tracking data...")
    import json

    tracking_data = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": len(frames),
        "ocr_model": stats['ocr_model'],
        "unique_players": stats['unique_jerseys'],
        "jersey_numbers": list(set(confirmed_numbers.values())),
        "tracker_to_jersey": {str(k): v for k, v in confirmed_numbers.items()},
        "pose_guided_crops": stats['pose_success'],
        "fallback_crops": stats['fallback'],
        "frames": {str(k): v for k, v in all_tracked.items()}
    }

    tracking_path = os.path.join(output_dir, "tracking_data.json")
    with open(tracking_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    print(f"    Tracking data saved: {tracking_path}")

    return {
        "video_path": output_video_path,
        "tracking_data_path": tracking_path,
        "unique_players": stats['unique_jerseys'],
        "jersey_numbers": list(set(confirmed_numbers.values())),
        "frames_processed": len(frames),
        "pose_guided_crops": stats['pose_success'],
        "fallback_crops": stats['fallback'],
    }


if __name__ == "__main__":
    import os
    # Auto-detect base path
    base_path = os.path.expanduser("~/swishvision")
    if not os.path.exists(base_path):
        base_path = "/workspace/swishvision"  # Fallback for RunPod

    results = process_video_jersey_numbers(
        video_path=f"{base_path}/test_videos/test_game.mp4",
        output_dir=f"{base_path}/outputs/jersey_pose_test",
        max_seconds=99999.0,
        ocr_interval=3,
        pose_model="yolov8x-pose-640"
    )
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"Video: {results['video_path']}")
    print(f"Unique Players: {results['unique_players']}")
    print(f"Jersey Numbers: {results['jersey_numbers']}")
    print(f"Pose-guided crops: {results['pose_guided_crops']}")
    print(f"Fallback crops: {results['fallback_crops']}")
