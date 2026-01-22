import os
import cv2
import numpy as np
from inference import get_model
from dotenv import load_dotenv

load_dotenv()

class CourtDetector:
    def __init__(self):
        self.model = get_model(
            model_id="basketball-court-detection-2/19",
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
        self.confidence = 0.3
        self.anchor_confidence = 0.5
    
    def detect_keypoints(self, frame: np.ndarray) -> dict:
        result = self.model.infer(frame, confidence=self.confidence)[0]
        
        # Keypoints are nested inside predictions[0].keypoints
        if not hasattr(result, 'predictions') or not result.predictions:
            return {
                "keypoints": None,
                "confidence": None,
                "mask": None,
                "count": 0
            }
        
        # Get keypoints from the first prediction (the court detection)
        prediction = result.predictions[0]
        
        if not hasattr(prediction, 'keypoints') or not prediction.keypoints:
            return {
                "keypoints": None,
                "confidence": None,
                "mask": None,
                "count": 0
            }
        
        keypoints = []
        confidences = []
        
        for kp in prediction.keypoints:
            keypoints.append([kp.x, kp.y])
            confidences.append(kp.confidence)
        
        keypoints = np.array(keypoints)
        confidences = np.array(confidences)
        mask = confidences >= self.anchor_confidence
        
        return {
            "keypoints": keypoints,
            "confidence": confidences,
            "mask": mask,
            "count": int(np.sum(mask))
        }
    
    def draw_keypoints(self, frame: np.ndarray, kp_result: dict, high_confidence_only: bool = True) -> np.ndarray:
        annotated = frame.copy()
        
        if kp_result["keypoints"] is None:
            return annotated
        
        keypoints = kp_result["keypoints"]
        confidences = kp_result["confidence"]
        mask = kp_result["mask"]
        
        for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
            x, y = int(kp[0]), int(kp[1])
            
            if x <= 0 or y <= 0:
                continue
            
            if high_confidence_only and not mask[i]:
                continue
            
            if mask[i]:
                color = (0, 255, 0)  # Green for high confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            cv2.circle(annotated, (x, y), 8, color, -1)
            cv2.putText(annotated, str(i), (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def get_court_hull(self, kp_result: dict) -> np.ndarray:
        """Get convex hull from high-confidence keypoints."""
        if kp_result["keypoints"] is None:
            return None
        
        keypoints = kp_result["keypoints"]
        mask = kp_result["mask"]
        
        # Filter to high-confidence points with valid coordinates
        valid_points = []
        for i, (kp, m) in enumerate(zip(keypoints, mask)):
            if m and kp[0] > 0 and kp[1] > 0:
                valid_points.append(kp)
        
        if len(valid_points) < 4:
            return None
        
        valid_points = np.array(valid_points, dtype=np.float32)
        hull = cv2.convexHull(valid_points)
        
        return hull
    
    def process_video(self, video_path: str, output_dir: str, num_frames: int = 3) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
        
        video_output_path = os.path.join(output_dir, "courtdetection_video_01.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        saved_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            kp_result = self.detect_keypoints(frame)
            annotated = self.draw_keypoints(frame, kp_result, high_confidence_only=True)
            
            video_writer.write(annotated)
            
            if frame_idx in frame_indices:
                frame_num = len(saved_frames) + 1
                frame_filename = f"courtdetection_frame_{frame_num:02d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, annotated)
                
                saved_frames.append({
                    "frame_number": frame_num,
                    "video_frame_index": frame_idx,
                    "keypoints_detected": kp_result["count"],
                    "output_path": frame_path
                })
            
            frame_idx += 1
        
        cap.release()
        video_writer.release()
        
        return {
            "video_path": video_path,
            "output_video": video_output_path,
            "frames_processed": frame_idx,
            "saved_frames": saved_frames
        }
    
    def get_court_keypoints(self) -> np.ndarray:
        """Reference court keypoints in real-world coordinates (feet)."""
        return np.array([
            [0, 0],       # Center court
            [-47, 25],    # Corner
            [-47, -25],
            [47, 25],
            [47, -25],
            [-47, 8],     # Paint corners
            [-47, -8],
            [-28, 8],
            [-28, -8],
            [-28, 0],
            [47, 8],
            [47, -8],
            [28, 8],
            [28, -8],
            [28, 0],
            [-22, 25],    # 3-point line
            [-22, -25],
            [22, 25],
            [22, -25],
            [-23.75, 8],
            [-23.75, -8],
            [23.75, 8],
            [23.75, -8],
            [0, 25],      # Half court
            [0, -25],
            [0, 6],
            [0, -6],
            [-47, 0],     # Baseline center
            [47, 0],
            [-28, 6],     # Free throw
            [-28, -6],
            [28, 6],
            [28, -6],
        ])