"""
Jersey Number Detector - Based on Roboflow Reference Implementation
Uses:
- RF-DETR to detect "number" class (class_id=2)
- Roboflow's fine-tuned SmolVLM2 OCR model
- ConsecutiveValueTracker for 3-consecutive validation
- Mask IoS matching to link numbers to players
"""
import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv
import supervision as sv
from inference import get_model

load_dotenv()

# Import ConsecutiveValueTracker from sports library
try:
    from sports import ConsecutiveValueTracker
    HAS_SPORTS_TRACKER = True
except ImportError:
    HAS_SPORTS_TRACKER = False
    print("Warning: sports.ConsecutiveValueTracker not available, using built-in")


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


def coords_above_threshold(matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Return (row, col) pairs where value > threshold, sorted descending"""
    rows, cols = np.where(matrix > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    pairs.sort(key=lambda rc: matrix[rc[0], rc[1]], reverse=True)
    return pairs


class JerseyNumberDetector:
    """
    Detects jersey numbers using Roboflow's approach:
    1. Detect "number" class boxes with RF-DETR
    2. Run OCR on number crops
    3. Match numbers to players using mask IoS
    4. Validate with 3 consecutive same reads
    """
    
    def __init__(self, device: str = None):
        # Device setup
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"JerseyNumberDetector using device: {self.device}")
        
        # Load models
        print("Loading player detection model...")
        self.player_model = get_model(
            model_id="basketball-player-detection-3-ycjdo/4",
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
        
        print("Loading number OCR model...")
        self.number_ocr_model = get_model(
            model_id="basketball-jersey-numbers-ocr/7",
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
        
        # Class IDs from the model
        self.NUMBER_CLASS_ID = 2
        self.PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]
        
        # Detection settings
        self.confidence = 0.4
        self.iou_threshold = 0.9
        
        # Number validator (3 consecutive same reads to confirm)
        if HAS_SPORTS_TRACKER:
            self.number_validator = ConsecutiveValueTracker(n_consecutive=3)
        else:
            self.number_validator = SimpleConsecutiveTracker(n_consecutive=3)
        
        # Store confirmed numbers
        self.confirmed_numbers: Dict[int, str] = {}
    
    def detect_numbers_in_frame(
        self,
        frame: np.ndarray,
        player_detections: sv.Detections,
    ) -> Dict[int, str]:
        """
        Detect and recognize jersey numbers, match to players.
        
        Args:
            frame: Video frame
            player_detections: Player detections with tracker_id and mask
            
        Returns:
            Dict mapping tracker_id -> jersey number string
        """
        import time
        
        frame_h, frame_w = frame.shape[:2]
        
        # Detect number boxes using RF-DETR
        t0 = time.time()
        result = self.player_model.infer(
            frame,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold
        )[0]
        all_detections = sv.Detections.from_inference(result)
        print(f"        Detection: {time.time()-t0:.2f}s")
        
        # Filter to only "number" class
        number_detections = all_detections[all_detections.class_id == self.NUMBER_CLASS_ID]
        print(f"        Found {len(number_detections)} number boxes")
        
        if len(number_detections) == 0:
            return {}
        
        # Create masks for number boxes
        number_detections.mask = sv.xyxy_to_mask(
            boxes=number_detections.xyxy,
            resolution_wh=(frame_w, frame_h)
        )
        
        # Crop and run OCR on each number
        number_crops = [
            sv.crop_image(frame, xyxy)
            for xyxy in sv.clip_boxes(
                sv.pad_boxes(xyxy=number_detections.xyxy, px=10, py=10),
                (frame_w, frame_h)
            )
        ]
        
        # Run OCR
        t0 = time.time()
        numbers = []
        for i, crop in enumerate(number_crops):
            try:
                result = self.number_ocr_model.predict(crop, "Read the number.")[0]
                numbers.append(str(result) if result else None)
                print(f"        OCR {i+1}/{len(number_crops)}: {result}")
            except Exception as e:
                print(f"        OCR {i+1}/{len(number_crops)}: ERROR - {e}")
                numbers.append(None)
        print(f"        OCR total: {time.time()-t0:.2f}s")
        
        # Ensure player detections have masks
        if player_detections.mask is None:
            # Create masks from boxes
            player_detections.mask = sv.xyxy_to_mask(
                boxes=player_detections.xyxy,
                resolution_wh=(frame_w, frame_h)
            )
        
        # Match numbers to players using mask IoS
        iou = sv.mask_iou_batch(
            masks_true=player_detections.mask,
            masks_detection=number_detections.mask,
            overlap_metric=sv.OverlapMetric.IOS
        )
        
        pairs = coords_above_threshold(iou, 0.9)
        
        if not pairs:
            return {}
        
        # Extract matched pairs
        player_idx, number_idx = zip(*pairs)
        matched_tracker_ids = [player_detections.tracker_id[i] for i in player_idx]
        matched_numbers = [numbers[i] for i in number_idx if numbers[i] is not None]
        
        # Filter out None values
        valid_pairs = [
            (tid, num) for tid, num in zip(matched_tracker_ids, [numbers[i] for i in number_idx])
            if num is not None
        ]
        
        if not valid_pairs:
            return {}
        
        tracker_ids_list = [p[0] for p in valid_pairs]
        numbers_list = [p[1] for p in valid_pairs]
        
        # Update validator
        self.number_validator.update(tracker_ids=tracker_ids_list, values=numbers_list)
        
        # Get validated numbers
        if HAS_SPORTS_TRACKER:
            validated = self.number_validator.get_validated(tracker_ids=player_detections.tracker_id)
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
        Simplified interface for player_tracker.py compatibility.
        """
        # Create sv.Detections
        detections = sv.Detections(
            xyxy=np.array(player_boxes),
            tracker_id=np.array(tracker_ids).astype(int),
            mask=player_masks
        )
        
        return self.detect_numbers_in_frame(frame, detections)
    
    def get_confirmed_numbers(self) -> Dict[int, str]:
        return self.confirmed_numbers.copy()
    
    def get_confirmed_count(self) -> int:
        return len(set(self.confirmed_numbers.values()))
    
    def clear_cache(self):
        self.confirmed_numbers.clear()
        if HAS_SPORTS_TRACKER:
            self.number_validator = ConsecutiveValueTracker(n_consecutive=3)
        else:
            self.number_validator = SimpleConsecutiveTracker(n_consecutive=3)


def process_video_jersey_numbers(
    video_path: str,
    output_dir: str,
    max_seconds: float = 99999.0,
    ocr_interval: int = 5,  # Run OCR every N frames (reference uses 5)
) -> Dict:
    """
    Process video and detect jersey numbers using the reference approach.
    """
    print("="*60)
    print("Jersey Number Detection - Roboflow Reference Approach")
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
    print("\n[2] Initializing jersey detector...")
    jersey_detector = JerseyNumberDetector()
    
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
        
        # Assign tracker IDs (simple approach - first frame gets sequential IDs)
        if frame_idx == 0:
            detections.tracker_id = np.arange(1, len(detections) + 1)
            tracker_id_counter = len(detections) + 1
        else:
            # Simple IoU matching to previous frame
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
            print(f"    Frame {frame_idx}: Running OCR...")
            
            # Create masks from boxes for IoS matching
            detections.mask = sv.xyxy_to_mask(
                boxes=detections.xyxy,
                resolution_wh=(width, height)
            )
            
            jersey_detector.detect_numbers_in_frame(frame, detections)
            
            confirmed = jersey_detector.get_confirmed_numbers()
            unique_count = jersey_detector.get_confirmed_count()
            print(f"    Frame {frame_idx}: {unique_count} unique players confirmed")
        
        # Progress every 10 frames
        if frame_idx % 10 == 0:
            print(f"    Processed {frame_idx}/{len(frames)} frames")
    
    # Final results
    confirmed_numbers = jersey_detector.get_confirmed_numbers()
    unique_count = jersey_detector.get_confirmed_count()
    
    print(f"\n[4] Results:")
    print(f"    Unique players confirmed: {unique_count}")
    print(f"    Jersey numbers: {list(set(confirmed_numbers.values()))}")
    
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
                # Only draw if confirmed
                jersey_num = confirmed_numbers.get(tid, None)
                if jersey_num is None:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in box]
                color = colors[tid % len(colors)]
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"#{jersey_num}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
                cv2.putText(annotated, lab el, (x1 + 3, y1 - 5),
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
        "unique_players": unique_count,
        "jersey_numbers": list(set(confirmed_numbers.values())),
        "tracker_to_jersey": {str(k): v for k, v in confirmed_numbers.items()},
        "frames": {str(k): v for k, v in all_tracked.items()}
    }
    
    tracking_path = os.path.join(output_dir, "tracking_data.json")
    with open(tracking_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    print(f"    Tracking data saved: {tracking_path}")
    
    return {
        "video_path": output_video_path,
        "tracking_data_path": tracking_path,
        "unique_players": unique_count,
        "jersey_numbers": list(set(confirmed_numbers.values())),
        "frames_processed": len(frames),
    }


if __name__ == "__main__":
    results = process_video_jersey_numbers(
        video_path="/workspace/swishvision/test_videos/test_game.mp4",
        output_dir="/workspace/swishvision/outputs/jersey_test",
        max_seconds=99999.0,  # Full video
        ocr_interval=1,  # OCR every 5 frames (same as reference)
    )
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"Video: {results['video_path']}")
    print(f"Unique Players: {results['unique_players']}")
    print(f"Jersey Numbers: {results['jersey_numbers']}")