import os
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
from app.ml.court_detector import CourtDetector

load_dotenv()


class CombinedDetector:
    """
    Combines RF-DETR (basketball-specific) with RTMDet (fast general person detection)
    to catch players that RF-DETR might miss.

    RTMDet variants:
    - rtmdet-tiny: 40.5% AP at 1020+ FPS
    - rtmdet-small: 44.6% AP at 819 FPS
    - rtmdet-medium: 48.8% AP
    - rtmdet-large: 51.2% AP at 300+ FPS
    - rtmdet-x: 52.8% AP (extra-large)
    """

    def __init__(self, rf_detr_confidence=0.25, rtmdet_confidence=0.3,
                 rtmdet_variant="rtmdet-large"):
        # RF-DETR: basketball-specific model
        self.rf_detr = get_model(
            model_id="basketball-player-detection-3-ycjdo/4",
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
        self.rf_detr_confidence = rf_detr_confidence
        self.rf_detr_player_class_ids = [3]
        self.rf_detr_referee_class_ids = [8]

        # RTMDet: fast general person detection (COCO class 0 = person)
        self.rtmdet = None
        self.rtmdet_variant = rtmdet_variant
        self.rtmdet_confidence = rtmdet_confidence
        self.rtmdet_person_class_id = 0  # COCO person class

        self._load_rtmdet()

        # Court detector for filtering on-court detections
        self.court_detector = CourtDetector()
        self._court_hull = None  # Cached court hull

        self.colors = {
            "rf_detr_player": (0, 255, 0),      # Green
            "rf_detr_referee": (128, 128, 128),  # Gray
            "rtmdet_person": (255, 165, 0),      # Orange (for RTMDet-only detections)
        }

    def _load_rtmdet(self):
        """Load RTMDet model from mmdetection."""
        try:
            from mmdet.apis import init_detector, inference_detector
            from mmdet.utils import register_all_modules

            register_all_modules()

            # RTMDet config and checkpoint paths
            config_map = {
                "rtmdet-tiny": "rtmdet_tiny_8xb32-300e_coco",
                "rtmdet-small": "rtmdet_s_8xb32-300e_coco",
                "rtmdet-medium": "rtmdet_m_8xb32-300e_coco",
                "rtmdet-large": "rtmdet_l_8xb32-300e_coco",
                "rtmdet-x": "rtmdet_x_8xb32-300e_coco",
            }

            config_name = config_map.get(self.rtmdet_variant, "rtmdet_l_8xb32-300e_coco")

            # Try to load from mmdetection model zoo
            self.rtmdet = init_detector(
                f"configs/rtmdet/{config_name}.py",
                f"https://download.openmmlab.com/mmdetection/v3.0/rtmdet/{config_name}/{config_name}.pth",
                device="cuda:0"
            )
            self._use_mmdet = True
            print(f"Loaded RTMDet ({self.rtmdet_variant}) via MMDetection")

        except Exception as e:
            print(f"MMDetection not available ({e}), falling back to YOLO")
            # Fallback to YOLOv8 via Roboflow if MMDetection not installed
            self.rtmdet = get_model(
                model_id="yolov8x-640",
                api_key=os.getenv("ROBOFLOW_API_KEY")
            )
            self._use_mmdet = False
            print("Using YOLOv8x as fallback for person detection")

    def _get_court_hull(self, frame: np.ndarray):
        """Get or compute the court convex hull for filtering."""
        if self._court_hull is not None:
            return self._court_hull

        kp_result = self.court_detector.detect_keypoints(frame)

        if kp_result["keypoints"] is None or kp_result["count"] < 4:
            return None

        keypoints = kp_result["keypoints"]
        mask = kp_result["mask"]
        valid_points = keypoints[mask]

        if len(valid_points) < 4:
            return None

        hull = cv2.convexHull(valid_points.astype(np.float32))

        # Expand hull slightly to include players near edges
        center = np.mean(hull.reshape(-1, 2), axis=0)
        expanded_hull = []
        for point in hull.reshape(-1, 2):
            direction = point - center
            expanded_point = point + direction * 0.05
            expanded_hull.append(expanded_point)
        self._court_hull = np.array(expanded_hull, dtype=np.float32).reshape(-1, 1, 2)

        return self._court_hull

    def filter_on_court(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Filter detections to only include those on the court."""
        if len(detections) == 0:
            return detections

        hull = self._get_court_hull(frame)
        if hull is None:
            return detections

        on_court_mask = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            # Use foot position (bottom center of bbox)
            foot_x = (x1 + x2) / 2
            foot_y = y2

            result = cv2.pointPolygonTest(hull, (foot_x, foot_y), False)
            on_court = result >= 0
            on_court_mask.append(on_court)

        return detections[np.array(on_court_mask)]

    def detect_rf_detr(self, frame: np.ndarray) -> sv.Detections:
        """Run RF-DETR basketball detection."""
        result = self.rf_detr.infer(frame, confidence=self.rf_detr_confidence)[0]
        detections = sv.Detections.from_inference(result)

        player_mask = np.isin(detections.class_id, self.rf_detr_player_class_ids)
        referee_mask = np.isin(detections.class_id, self.rf_detr_referee_class_ids)
        combined_mask = player_mask | referee_mask

        return detections[combined_mask]

    def detect_rtmdet(self, frame: np.ndarray) -> sv.Detections:
        """Run RTMDet/YOLO person detection."""
        if self._use_mmdet:
            from mmdet.apis import inference_detector
            result = inference_detector(self.rtmdet, frame)

            # Extract person class predictions (class 0)
            pred_instances = result.pred_instances
            person_mask = pred_instances.labels == self.rtmdet_person_class_id
            conf_mask = pred_instances.scores >= self.rtmdet_confidence
            mask = person_mask & conf_mask

            boxes = pred_instances.bboxes[mask].cpu().numpy()
            scores = pred_instances.scores[mask].cpu().numpy()

            if len(boxes) == 0:
                return sv.Detections.empty()

            return sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=np.zeros(len(boxes), dtype=int)
            )
        else:
            # Fallback YOLO path
            result = self.rtmdet.infer(frame, confidence=self.rtmdet_confidence)[0]
            detections = sv.Detections.from_inference(result)
            person_mask = detections.class_id == self.rtmdet_person_class_id
            return detections[person_mask]

    def merge_detections(self, rf_detr_dets: sv.Detections, rtmdet_dets: sv.Detections,
                         iou_threshold: float = 0.5) -> sv.Detections:
        """
        Merge detections from both models.
        - Keep all RF-DETR detections (they're basketball-specific)
        - Add RTMDet detections that don't overlap with RF-DETR (new players)
        """
        if len(rf_detr_dets) == 0:
            if len(rtmdet_dets) == 0:
                return sv.Detections.empty()
            # No RF-DETR detections, use all RTMDet
            rtmdet_dets.class_id = np.full(len(rtmdet_dets), 3)  # Mark as player
            rtmdet_dets.data = {"source": np.ones(len(rtmdet_dets), dtype=int)}
            return rtmdet_dets

        if len(rtmdet_dets) == 0:
            # No RTMDet detections, use all RF-DETR
            rf_detr_dets.data = {"source": np.zeros(len(rf_detr_dets), dtype=int)}
            return rf_detr_dets

        # Calculate IoU between RTMDet and RF-DETR detections
        iou_matrix = sv.box_iou_batch(rtmdet_dets.xyxy, rf_detr_dets.xyxy)

        # Find RTMDet detections that don't overlap with any RF-DETR detection
        max_iou_per_rtmdet = np.max(iou_matrix, axis=1)
        new_detections_mask = max_iou_per_rtmdet < iou_threshold

        new_rtmdet_dets = rtmdet_dets[new_detections_mask]

        if len(new_rtmdet_dets) == 0:
            rf_detr_dets.data = {"source": np.zeros(len(rf_detr_dets), dtype=int)}
            return rf_detr_dets

        # Merge: RF-DETR detections + new RTMDet detections
        merged_xyxy = np.vstack([rf_detr_dets.xyxy, new_rtmdet_dets.xyxy])
        merged_confidence = np.concatenate([rf_detr_dets.confidence, new_rtmdet_dets.confidence])

        # Use class_id 3 (player) for RTMDet detections
        merged_class_id = np.concatenate([
            rf_detr_dets.class_id,
            np.full(len(new_rtmdet_dets), 3)  # Mark as player
        ])

        # Track which detector found each detection (for visualization)
        rf_detr_sources = np.zeros(len(rf_detr_dets), dtype=int)  # 0 = RF-DETR
        rtmdet_sources = np.ones(len(new_rtmdet_dets), dtype=int)  # 1 = RTMDet
        sources = np.concatenate([rf_detr_sources, rtmdet_sources])

        merged = sv.Detections(
            xyxy=merged_xyxy,
            confidence=merged_confidence,
            class_id=merged_class_id,
        )
        merged.data = {"source": sources}

        return merged

    def detect(self, frame: np.ndarray, filter_court: bool = True) -> sv.Detections:
        """Run both detectors and merge results."""
        rf_detr_dets = self.detect_rf_detr(frame)
        rtmdet_dets = self.detect_rtmdet(frame)

        # Filter both to on-court only for fair comparison
        if filter_court:
            rf_detr_dets = self.filter_on_court(rf_detr_dets, frame)
            rtmdet_dets = self.filter_on_court(rtmdet_dets, frame)

        return self.merge_detections(rf_detr_dets, rtmdet_dets)

    def draw_detections(self, frame: np.ndarray, detections: sv.Detections,
                        show_source: bool = True) -> np.ndarray:
        """Draw detections with color-coding by source."""
        annotated = frame.copy()

        sources = detections.data.get("source", np.zeros(len(detections))) if detections.data else np.zeros(len(detections))

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            confidence = detections.confidence[i]
            source = sources[i] if i < len(sources) else 0

            if source == 0:
                color = self.colors["rf_detr_player"]
                label = f"RF-DETR {confidence:.2f}"
            else:
                color = self.colors["rtmdet_person"]
                label = f"RTMDet {confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated

    def draw_comparison(self, frame: np.ndarray, rf_detr_dets: sv.Detections,
                        rtmdet_dets: sv.Detections, merged_dets: sv.Detections) -> np.ndarray:
        """Create a side-by-side comparison image."""
        h, w = frame.shape[:2]

        # Create 3 annotated versions
        rf_detr_frame = frame.copy()
        rtmdet_frame = frame.copy()
        merged_frame = frame.copy()

        # Draw RF-DETR detections (green)
        for i in range(len(rf_detr_dets)):
            x1, y1, x2, y2 = rf_detr_dets.xyxy[i].astype(int)
            cv2.rectangle(rf_detr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rf_detr_frame, f"RF-DETR: {len(rf_detr_dets)} detected",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw RTMDet detections (orange)
        for i in range(len(rtmdet_dets)):
            x1, y1, x2, y2 = rtmdet_dets.xyxy[i].astype(int)
            cv2.rectangle(rtmdet_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(rtmdet_frame, f"RTMDet: {len(rtmdet_dets)} detected",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        # Draw merged detections with source colors
        merged_frame = self.draw_detections(merged_frame, merged_dets)
        cv2.putText(merged_frame, f"Combined: {len(merged_dets)} detected",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Stack horizontally
        comparison = np.hstack([rf_detr_frame, rtmdet_frame, merged_frame])

        return comparison

    def process_frames(self, video_path: str, output_dir: str,
                       frame_indices: list = None, num_frames: int = 3,
                       filter_court: bool = True) -> dict:
        """Process specific frames and save comparison images."""
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_indices is None:
            # Pick evenly spaced frames
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        results = []

        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Run both detectors
            rf_detr_dets = self.detect_rf_detr(frame)
            rtmdet_dets = self.detect_rtmdet(frame)

            # Filter to on-court only for fair comparison
            if filter_court:
                rf_detr_dets = self.filter_on_court(rf_detr_dets, frame)
                rtmdet_dets = self.filter_on_court(rtmdet_dets, frame)

            merged_dets = self.merge_detections(rf_detr_dets, rtmdet_dets)

            # Save individual detection frame
            annotated = self.draw_detections(frame, merged_dets)
            single_path = os.path.join(output_dir, f"detection_frame_{idx+1:02d}.jpg")
            cv2.imwrite(single_path, annotated)

            # Save comparison image
            comparison = self.draw_comparison(frame, rf_detr_dets, rtmdet_dets, merged_dets)
            comparison_path = os.path.join(output_dir, f"comparison_frame_{idx+1:02d}.jpg")
            cv2.imwrite(comparison_path, comparison)

            # Count sources in merged
            sources = merged_dets.data.get("source", np.zeros(len(merged_dets))) if merged_dets.data else np.zeros(len(merged_dets))
            rf_detr_count = int(np.sum(sources == 0))
            rtmdet_only_count = int(np.sum(sources == 1))

            results.append({
                "frame_index": frame_idx,
                "rf_detr_detections": len(rf_detr_dets),
                "rtmdet_detections": len(rtmdet_dets),
                "merged_total": len(merged_dets),
                "rtmdet_added": rtmdet_only_count,  # Players RTMDet found that RF-DETR missed
                "single_path": single_path,
                "comparison_path": comparison_path,
            })

            print(f"Frame {frame_idx}: RF-DETR={len(rf_detr_dets)}, RTMDet={len(rtmdet_dets)}, "
                  f"Combined={len(merged_dets)} (+{rtmdet_only_count} from RTMDet)")

        cap.release()

        return {
            "video_path": video_path,
            "output_dir": output_dir,
            "frames": results
        }

    def process_video(self, video_path: str, output_dir: str,
                      filter_court: bool = True, max_frames: int = None) -> dict:
        """Process entire video and output detection video."""
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        # Video writers for each output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        merged_video = cv2.VideoWriter(
            os.path.join(output_dir, "combined_detection.mp4"),
            fourcc, fps, (width, height)
        )
        comparison_video = cv2.VideoWriter(
            os.path.join(output_dir, "comparison_detection.mp4"),
            fourcc, fps, (width * 3, height)  # 3 panels side by side
        )

        stats = {
            "rf_detr_total": 0,
            "rtmdet_total": 0,
            "rtmdet_added_total": 0,
            "frames_processed": 0
        }

        from tqdm import tqdm
        for frame_idx in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Run both detectors
            rf_detr_dets = self.detect_rf_detr(frame)
            rtmdet_dets = self.detect_rtmdet(frame)

            # Filter to on-court only
            if filter_court:
                rf_detr_dets = self.filter_on_court(rf_detr_dets, frame)
                rtmdet_dets = self.filter_on_court(rtmdet_dets, frame)

            merged_dets = self.merge_detections(rf_detr_dets, rtmdet_dets)

            # Draw and write frames
            annotated = self.draw_detections(frame, merged_dets)
            merged_video.write(annotated)

            comparison = self.draw_comparison(frame, rf_detr_dets, rtmdet_dets, merged_dets)
            comparison_video.write(comparison)

            # Update stats
            sources = merged_dets.data.get("source", np.zeros(len(merged_dets))) if merged_dets.data else np.zeros(len(merged_dets))
            stats["rf_detr_total"] += len(rf_detr_dets)
            stats["rtmdet_total"] += len(rtmdet_dets)
            stats["rtmdet_added_total"] += int(np.sum(sources == 1))
            stats["frames_processed"] += 1

        cap.release()
        merged_video.release()
        comparison_video.release()

        print(f"\n=== Video Processing Complete ===")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Avg RF-DETR detections/frame: {stats['rf_detr_total'] / stats['frames_processed']:.1f}")
        print(f"Avg RTMDet detections/frame: {stats['rtmdet_total'] / stats['frames_processed']:.1f}")
        print(f"Avg players added by RTMDet: {stats['rtmdet_added_total'] / stats['frames_processed']:.1f}")

        return {
            "video_path": video_path,
            "output_dir": output_dir,
            "merged_video": os.path.join(output_dir, "combined_detection.mp4"),
            "comparison_video": os.path.join(output_dir, "comparison_detection.mp4"),
            "stats": stats
        }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Combined RF-DETR + RTMDet detector")
    parser.add_argument("video_path", nargs="?", default="../test_videos/test_game.mp4",
                        help="Path to input video")
    parser.add_argument("output_dir", nargs="?", default="output/combined_detection",
                        help="Output directory")
    parser.add_argument("--video", action="store_true",
                        help="Process entire video instead of just 3 frames")
    parser.add_argument("--frames", type=int, default=3,
                        help="Number of frames to sample (default: 3)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process in video mode")
    parser.add_argument("--no-court-filter", action="store_true",
                        help="Disable court filtering")

    args = parser.parse_args()

    detector = CombinedDetector()

    if args.video:
        # Process entire video
        results = detector.process_video(
            args.video_path,
            args.output_dir,
            filter_court=not args.no_court_filter,
            max_frames=args.max_frames
        )
        print(f"\nOutput videos saved to: {args.output_dir}")
    else:
        # Process sample frames
        results = detector.process_frames(
            args.video_path,
            args.output_dir,
            num_frames=args.frames,
            filter_court=not args.no_court_filter
        )

        print("\n=== Results ===")
        for frame in results["frames"]:
            print(f"Frame {frame['frame_index']}:")
            print(f"  RF-DETR: {frame['rf_detr_detections']}")
            print(f"  RTMDet: {frame['rtmdet_detections']}")
            print(f"  Combined: {frame['merged_total']} (+{frame['rtmdet_added']} from RTMDet)")
            print(f"  Saved: {frame['comparison_path']}")
