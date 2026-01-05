import os
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from dotenv import load_dotenv

load_dotenv()

class PlayerDetector:
    def __init__(self):
        self.model = get_model(
            model_id="basketball-player-detection-3-ycjdo/4",
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
        self.confidence = 0.25
        
        self.player_class_ids = [3]
        self.referee_class_ids = [8]
        
        self.colors = {
            "player": (0, 255, 0),
            "referee": (128, 128, 128)
        }
        
        self.court_detector = None
    
    def set_court_detector(self, court_detector):
        self.court_detector = court_detector
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.infer(frame, confidence=self.confidence)[0]
        detections = sv.Detections.from_inference(result)
        
        player_mask = np.isin(detections.class_id, self.player_class_ids)
        referee_mask = np.isin(detections.class_id, self.referee_class_ids)
        combined_mask = player_mask | referee_mask
        
        return detections[combined_mask]
    
    def filter_on_court(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        if self.court_detector is None:
            return detections
        
        kp_result = self.court_detector.detect_keypoints(frame)
        
        if kp_result["keypoints"] is None or kp_result["count"] < 4:
            return detections
        
        keypoints = kp_result["keypoints"]
        mask = kp_result["mask"]
        valid_points = keypoints[mask]
        
        if len(valid_points) < 4:
            return detections
        
        hull = cv2.convexHull(valid_points.astype(np.float32))
        
        center = np.mean(hull.reshape(-1, 2), axis=0)
        expanded_hull = []
        for point in hull.reshape(-1, 2):
            direction = point - center
            expanded_point = point + direction * 0.05
            expanded_hull.append(expanded_point)
        expanded_hull = np.array(expanded_hull, dtype=np.float32).reshape(-1, 1, 2)
        
        on_court_mask = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            foot_x = (x1 + x2) / 2
            foot_y = y2
            
            result = cv2.pointPolygonTest(expanded_hull, (foot_x, foot_y), False)
            on_court = result >= 0
            on_court_mask.append(on_court)
        
        return detections[np.array(on_court_mask)]
    
    def draw_detections(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated = frame.copy()
        
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            
            if class_id == 3:
                class_name = "player"
                color = self.colors["player"]
            else:
                class_name = "referee"
                color = self.colors["referee"]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def process_video(self, video_path: str, output_dir: str, num_frames: int = 3, filter_court: bool = False) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
        
        video_output_path = os.path.join(output_dir, "playerdetection_video_01.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        saved_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect(frame)
            
            if filter_court:
                detections = self.filter_on_court(detections, frame)
            
            annotated = self.draw_detections(frame, detections)
            
            video_writer.write(annotated)
            
            if frame_idx in frame_indices:
                frame_num = len(saved_frames) + 1
                frame_filename = f"playerdetection_frame_{frame_num:02d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, annotated)
                
                player_count = np.sum(detections.class_id == 3)
                referee_count = np.sum(detections.class_id == 8)
                
                saved_frames.append({
                    "frame_number": frame_num,
                    "video_frame_index": frame_idx,
                    "players": int(player_count),
                    "referees": int(referee_count),
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