import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv
from app.ml.court_detector import CourtDetector

load_dotenv()


class YOLOv11PlayerTracker:
    """
    Uses YOLOv11 for player and referee tracking with continuous detection
    every 15 frames to maintain tracking accuracy.
    """

    def __init__(self, model_path: str = "yolo11x.pt", confidence: float = 0.3):
        # Load YOLOv11 model
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class_id = 0  # COCO person class

        # Court detector for filtering on-court detections
        self.court_detector = CourtDetector()
        self._court_hull = None

        # Tracking state
        self.tracker = sv.ByteTrack()

        self.colors = {
            "player": (0, 255, 0),      # Green
            "referee": (128, 128, 128),  # Gray (if we can distinguish)
        }

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

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLOv11 detection on a frame."""
        results = self.model(frame, conf=self.confidence, verbose=False)[0]

        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filter to only person class
        if len(detections) > 0:
            person_mask = detections.class_id == self.person_class_id
            detections = detections[person_mask]

        return detections

    def draw_detections(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw detections with tracking IDs."""
        annotated = frame.copy()

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            confidence = detections.confidence[i]

            # Get tracker ID if available
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None

            color = self.colors["player"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label with tracker ID
            if tracker_id is not None:
                label = f"ID:{tracker_id} {confidence:.2f}"
            else:
                label = f"{confidence:.2f}"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add count
        cv2.putText(annotated, f"Players: {len(detections)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated

    def process_video(self, video_path: str, output_dir: str,
                      filter_court: bool = True,
                      detection_interval: int = 15,
                      max_seconds: float = 10.0) -> dict:
        """
        Process video with YOLOv11 tracking.

        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            filter_court: Whether to filter detections to on-court only
            detection_interval: Run detection every N frames (default 15)
            max_seconds: Maximum seconds of video to process (default 10)
        """
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Limit to max_seconds
        max_frames = int(fps * max_seconds)
        total_frames = min(total_frames, max_frames)

        print(f"Processing {total_frames} frames ({max_seconds}s at {fps:.1f} FPS)")
        print(f"Detection interval: every {detection_interval} frames")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(
            os.path.join(output_dir, "yolov11_tracking.mp4"),
            fourcc, fps, (width, height)
        )

        stats = {
            "total_detections": 0,
            "detection_frames": 0,
            "frames_processed": 0
        }

        # Store last detections for frames between detection intervals
        last_detections = None

        from tqdm import tqdm
        for frame_idx in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection every detection_interval frames
            if frame_idx % detection_interval == 0:
                detections = self.detect(frame)

                # Filter to on-court only
                if filter_court:
                    detections = self.filter_on_court(detections, frame)

                # Update tracker with new detections
                detections = self.tracker.update_with_detections(detections)

                last_detections = detections
                stats["detection_frames"] += 1
                stats["total_detections"] += len(detections)
            else:
                # Use tracking to propagate between detection frames
                if last_detections is not None and len(last_detections) > 0:
                    # Run detection but only for tracking update
                    detections = self.detect(frame)
                    if filter_court:
                        detections = self.filter_on_court(detections, frame)
                    detections = self.tracker.update_with_detections(detections)
                    last_detections = detections
                else:
                    detections = sv.Detections.empty()

            # Draw and write frame
            annotated = self.draw_detections(frame, detections if detections is not None else sv.Detections.empty())

            # Add frame info
            cv2.putText(annotated, f"Frame: {frame_idx}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            is_detection_frame = frame_idx % detection_interval == 0
            status = "DETECT" if is_detection_frame else "TRACK"
            color = (0, 255, 0) if is_detection_frame else (255, 165, 0)
            cv2.putText(annotated, status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            output_video.write(annotated)
            stats["frames_processed"] += 1

        cap.release()
        output_video.release()

        print(f"\n=== Processing Complete ===")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Detection frames: {stats['detection_frames']}")
        print(f"Avg detections/detection-frame: {stats['total_detections'] / max(stats['detection_frames'], 1):.1f}")

        return {
            "video_path": video_path,
            "output_dir": output_dir,
            "output_video": os.path.join(output_dir, "yolov11_tracking.mp4"),
            "stats": stats
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv11 Player Tracker")
    parser.add_argument("video_path", nargs="?", default="../test_videos/test_game.mp4",
                        help="Path to input video")
    parser.add_argument("output_dir", nargs="?", default="output/yolov11_tracking",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="yolo11x.pt",
                        help="YOLOv11 model path (default: yolo11x.pt)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--detection-interval", type=int, default=15,
                        help="Run detection every N frames (default: 15)")
    parser.add_argument("--max-seconds", type=float, default=10.0,
                        help="Max seconds of video to process (default: 10)")
    parser.add_argument("--no-court-filter", action="store_true",
                        help="Disable court filtering")

    args = parser.parse_args()

    tracker = YOLOv11PlayerTracker(model_path=args.model, confidence=args.confidence)

    results = tracker.process_video(
        args.video_path,
        args.output_dir,
        filter_court=not args.no_court_filter,
        detection_interval=args.detection_interval,
        max_seconds=args.max_seconds
    )

    print(f"\nOutput video: {results['output_video']}")
