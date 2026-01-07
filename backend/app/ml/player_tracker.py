"""
Player tracking using SAM2 for video segmentation.
Includes team classification, tactical view, and jersey number detection.
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


class PlayerTracker:
    def __init__(
        self,
        sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
        sam2_config: str = "sam2.1_hiera_l",
        device: str = None,
        enable_jersey_numbers: bool = True,
    ):
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.enable_jersey_numbers = enable_jersey_numbers
        
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
        
        self.jersey_detector = None
        if enable_jersey_numbers:
            try:
                from app.ml.jersey_number_detector import JerseyNumberDetector
                self.jersey_detector = JerseyNumberDetector()
                print("Jersey number detection enabled")
            except Exception as e:
                print(f"Jersey number detection disabled: {e}")
    
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
        detect_jersey_numbers: bool = True,
    ) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        use_jersey_detection = detect_jersey_numbers and self.jersey_detector is not None
        
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
                    info['team_name'] = 'Team A'
            
            jersey_numbers = {}
            if use_jersey_detection:
                print("Detecting jersey numbers...")
                self.jersey_detector.clear_cache()
                
                for fidx in range(0, frame_count, max(1, frame_count // 10)):
                    if fidx not in video_segments:
                        continue
                    
                    frame = frames[fidx]
                    boxes, obj_ids_list = [], []
                    
                    for obj_id, mask in video_segments[fidx].items():
                        if obj_id not in tracking_info or tracking_info[obj_id]['class'] == 'referee':
                            continue
                        box = mask_to_box(mask)
                        if box is not None:
                            boxes.append(box)
                            obj_ids_list.append(obj_id)
                    
                    if not boxes:
                        continue
                    
                    player_dets = sv.Detections(xyxy=np.array(boxes), class_id=np.array([3] * len(boxes)))
                    frame_numbers = self.jersey_detector.process_frame(frame, player_dets, tracker_ids=np.array(obj_ids_list))
                    jersey_numbers.update(frame_numbers)
                
                for obj_id, number in jersey_numbers.items():
                    if obj_id in tracking_info:
                        tracking_info[obj_id]['jersey_number'] = number
                
                print(f"  Detected {len(jersey_numbers)} jersey numbers")
            
            print("Initializing tactical view...")
            tactical_view = TacticalView()
            
            def annotate_frame(frame, frame_idx):
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
                        
                        jersey_num = info.get('jersey_number')
                        team_name = info.get('team_name', 'Player')
                        label = f"#{jersey_num} {team_name}" if jersey_num else f"#{obj_id} {team_name}"
                        
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                if player_positions:
                    ta = {oid: tracking_info[oid].get('team', 0) for oid in player_positions}
                    tc = {0: team_classifier.get_team_color(0), 1: team_classifier.get_team_color(1), -1: (0, 255, 255)}
                    # Get jersey numbers for tactical view (use obj_id as fallback)
                    jn = {oid: tracking_info[oid].get('jersey_number', str(oid)) for oid in player_positions}
                    # Use render_with_numbers for numbered circles on tactical view
                    tactical = tactical_view.render_with_numbers(player_positions, (height, width), ta, tc, jn)
                    annotated = create_combined_view(annotated, tactical)
                
                return annotated
            
            sample_indices = [int(i * (frame_count - 1) / (num_sample_frames - 1)) for i in range(num_sample_frames)]
            
            print("Generating sample frames...")
            sample_frames = []
            for idx, frame_idx in enumerate(sample_indices):
                annotated = annotate_frame(frames[frame_idx], frame_idx)
                path = os.path.join(output_dir, f"tracking_frame_{idx+1:02d}.jpg")
                cv2.imwrite(path, annotated)
                sample_frames.append(path)
            print(f"  Saved {len(sample_frames)} frames")
            
            print("Generating video...")
            output_video_path = os.path.join(output_dir, "tracking_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            for frame_idx in range(frame_count):
                annotated = annotate_frame(frames[frame_idx], frame_idx)
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