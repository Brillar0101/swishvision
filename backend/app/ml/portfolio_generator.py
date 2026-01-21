"""
Portfolio Video Generator for SwishVision.
Generates separate videos showing each pipeline stage for demonstration purposes.

Stages:
1. Raw Detection - Bounding boxes from RF-DETR player detector
2. Segmentation - SAM2 player masks
3. Team Classification - Color-coded teams
4. Jersey Detection - Numbers and player names
5. Tactical View - 2D court minimap
6. Combined - Full pipeline output
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import supervision as sv

from app.ml.tactical_view import TacticalView, draw_court, create_combined_view
from app.ml.team_rosters import TEAM_ROSTERS, TEAM_COLORS, get_player_name


def mask_to_box(mask):
    """Convert binary mask to bounding box."""
    mask_2d = mask.squeeze()
    ys, xs = np.where(mask_2d)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


class PortfolioGenerator:
    """
    Generates portfolio videos showing each stage of the SwishVision pipeline.
    """

    def __init__(
        self,
        team_colors: Dict[int, Tuple[int, int, int]] = None,
        team_names: Tuple[str, str] = ("Team A", "Team B"),
    ):
        self.team_colors = team_colors or {
            0: (0, 255, 0),    # Green
            1: (0, 0, 255),    # Red
            -1: (0, 255, 255), # Yellow (referees)
        }
        self.team_names = team_names
        self.tactical_view = TacticalView()

    def generate_stage_videos(
        self,
        frames: List[np.ndarray],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        tracking_info: Dict[int, Dict],
        detections_per_frame: Dict[int, sv.Detections],
        output_dir: str,
        fps: float = 30.0,
        smoothed_positions: List[Dict[int, Tuple[float, float]]] = None,
    ) -> Dict[str, str]:
        """
        Generate separate videos for each pipeline stage.

        Args:
            frames: List of video frames
            video_segments: Dict[frame_idx][obj_id] -> mask
            tracking_info: Dict[obj_id] -> {'team', 'team_name', 'jersey_number', 'player_name', ...}
            detections_per_frame: Dict[frame_idx] -> sv.Detections (raw detections)
            output_dir: Output directory for videos
            fps: Video frame rate
            smoothed_positions: Optional smoothed positions for tactical view

        Returns:
            Dict mapping stage name to video path
        """
        os.makedirs(output_dir, exist_ok=True)

        height, width = frames[0].shape[:2]
        frame_count = len(frames)

        output_paths = {}

        # Stage 1: Raw Detection
        print("Generating Stage 1: Raw Detection...")
        output_paths['detection'] = self._generate_detection_video(
            frames, detections_per_frame, output_dir, fps
        )

        # Stage 2: Segmentation
        print("Generating Stage 2: Segmentation...")
        output_paths['segmentation'] = self._generate_segmentation_video(
            frames, video_segments, output_dir, fps
        )

        # Stage 3: Team Classification
        print("Generating Stage 3: Team Classification...")
        output_paths['teams'] = self._generate_teams_video(
            frames, video_segments, tracking_info, output_dir, fps
        )

        # Stage 4: Jersey Detection
        print("Generating Stage 4: Jersey Detection...")
        output_paths['jersey'] = self._generate_jersey_video(
            frames, video_segments, tracking_info, output_dir, fps
        )

        # Stage 5: Tactical View Only
        print("Generating Stage 5: Tactical View...")
        output_paths['tactical'] = self._generate_tactical_video(
            frames, video_segments, tracking_info, output_dir, fps, smoothed_positions
        )

        # Stage 6: Combined View
        print("Generating Stage 6: Combined View...")
        output_paths['combined'] = self._generate_combined_video(
            frames, video_segments, tracking_info, output_dir, fps, smoothed_positions
        )

        print(f"Generated {len(output_paths)} portfolio videos")
        return output_paths

    def _get_video_writer(self, path: str, fps: float, width: int, height: int):
        """Create video writer."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(path, fourcc, fps, (width, height))

    def _generate_detection_video(
        self,
        frames: List[np.ndarray],
        detections_per_frame: Dict[int, sv.Detections],
        output_dir: str,
        fps: float,
    ) -> str:
        """Generate video showing raw bounding box detections."""
        output_path = os.path.join(output_dir, "stage1_detection.mp4")
        height, width = frames[0].shape[:2]
        writer = self._get_video_writer(output_path, fps, width, height)

        # Create annotators
        box_annotator = sv.BoxAnnotator(thickness=2)

        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            # Add stage label
            self._add_stage_label(annotated, "Stage 1: Player Detection (RF-DETR)")

            if frame_idx in detections_per_frame:
                detections = detections_per_frame[frame_idx]
                annotated = box_annotator.annotate(annotated, detections)

                # Add detection count
                count_text = f"Detected: {len(detections)} players"
                cv2.putText(annotated, count_text, (20, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(annotated)

        writer.release()
        return output_path

    def _generate_segmentation_video(
        self,
        frames: List[np.ndarray],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        output_dir: str,
        fps: float,
    ) -> str:
        """Generate video showing SAM2 segmentation masks."""
        output_path = os.path.join(output_dir, "stage2_segmentation.mp4")
        height, width = frames[0].shape[:2]
        writer = self._get_video_writer(output_path, fps, width, height)

        # Generate distinct colors for each object
        colors = self._generate_distinct_colors(20)

        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            # Add stage label
            self._add_stage_label(annotated, "Stage 2: Player Segmentation (SAM2)")

            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    mask_2d = mask.squeeze()
                    color = colors[obj_id % len(colors)]

                    # Draw colored mask
                    mask_colored = np.zeros_like(annotated)
                    mask_colored[mask_2d] = color
                    annotated = cv2.addWeighted(annotated, 1.0, mask_colored, 0.5, 0)

                    # Draw outline
                    contours, _ = cv2.findContours(
                        mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annotated, contours, -1, color, 2)

                    # Add object ID
                    box = mask_to_box(mask)
                    if box:
                        x1, y1 = int(box[0]), int(box[1])
                        cv2.putText(annotated, f"ID:{obj_id}", (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Add segment count
                count_text = f"Segments: {len(video_segments[frame_idx])}"
                cv2.putText(annotated, count_text, (20, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(annotated)

        writer.release()
        return output_path

    def _generate_teams_video(
        self,
        frames: List[np.ndarray],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        tracking_info: Dict[int, Dict],
        output_dir: str,
        fps: float,
    ) -> str:
        """Generate video showing team classification."""
        output_path = os.path.join(output_dir, "stage3_teams.mp4")
        height, width = frames[0].shape[:2]
        writer = self._get_video_writer(output_path, fps, width, height)

        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            # Add stage label
            self._add_stage_label(annotated, "Stage 3: Team Classification (SigLIP + K-means)")

            if frame_idx in video_segments:
                team_counts = {0: 0, 1: 0, -1: 0}

                for obj_id, mask in video_segments[frame_idx].items():
                    mask_2d = mask.squeeze()
                    info = tracking_info.get(obj_id, {})
                    team_id = info.get('team', 0)
                    team_name = info.get('team_name', 'Unknown')
                    color = self.team_colors.get(team_id, (128, 128, 128))

                    team_counts[team_id] = team_counts.get(team_id, 0) + 1

                    # Draw colored mask
                    mask_colored = np.zeros_like(annotated)
                    mask_colored[mask_2d] = color
                    annotated = cv2.addWeighted(annotated, 1.0, mask_colored, 0.4, 0)

                    # Draw bounding box
                    box = mask_to_box(mask)
                    if box:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                        # Add team label
                        label = team_name
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Add team counts legend
                y_offset = height - 80
                for team_id, count in team_counts.items():
                    if count > 0:
                        color = self.team_colors.get(team_id, (128, 128, 128))
                        if team_id == -1:
                            name = "Referees"
                        elif team_id < len(self.team_names):
                            name = self.team_names[team_id]
                        else:
                            name = f"Team {team_id}"
                        text = f"{name}: {count}"
                        cv2.putText(annotated, text, (20, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 25

            writer.write(annotated)

        writer.release()
        return output_path

    def _generate_jersey_video(
        self,
        frames: List[np.ndarray],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        tracking_info: Dict[int, Dict],
        output_dir: str,
        fps: float,
    ) -> str:
        """Generate video showing jersey number detection and player names."""
        output_path = os.path.join(output_dir, "stage4_jersey.mp4")
        height, width = frames[0].shape[:2]
        writer = self._get_video_writer(output_path, fps, width, height)

        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            # Add stage label
            self._add_stage_label(annotated, "Stage 4: Jersey Detection (RF-DETR + SmolVLM2 OCR)")

            if frame_idx in video_segments:
                identified_count = 0

                for obj_id, mask in video_segments[frame_idx].items():
                    mask_2d = mask.squeeze()
                    info = tracking_info.get(obj_id, {})
                    team_id = info.get('team', 0)
                    color = self.team_colors.get(team_id, (128, 128, 128))

                    jersey_number = info.get('jersey_number')
                    player_name = info.get('player_name')

                    if jersey_number:
                        identified_count += 1

                    # Draw colored mask
                    mask_colored = np.zeros_like(annotated)
                    mask_colored[mask_2d] = color
                    annotated = cv2.addWeighted(annotated, 1.0, mask_colored, 0.4, 0)

                    # Draw bounding box
                    box = mask_to_box(mask)
                    if box:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                        # Create label with jersey number and player name
                        if jersey_number and player_name:
                            label = f"#{jersey_number} {player_name}"
                        elif jersey_number:
                            label = f"#{jersey_number}"
                        else:
                            label = f"ID:{obj_id}"

                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Add identification stats
                total = len(video_segments[frame_idx])
                stats_text = f"Identified: {identified_count}/{total} players"
                cv2.putText(annotated, stats_text, (20, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(annotated)

        writer.release()
        return output_path

    def _generate_tactical_video(
        self,
        frames: List[np.ndarray],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        tracking_info: Dict[int, Dict],
        output_dir: str,
        fps: float,
        smoothed_positions: List[Dict[int, Tuple[float, float]]] = None,
    ) -> str:
        """Generate video showing tactical 2D court view (full screen)."""
        output_path = os.path.join(output_dir, "stage5_tactical.mp4")

        # Tactical view dimensions (16:9 aspect ratio scaled from court)
        tactical_width = 1280
        tactical_height = 720

        writer = self._get_video_writer(output_path, fps, tactical_width, tactical_height)

        for frame_idx, frame in enumerate(frames):
            # Build transformer from frame
            self.tactical_view.build_transformer(frame)

            # Get positions
            if smoothed_positions and frame_idx < len(smoothed_positions):
                positions = smoothed_positions[frame_idx]
            elif frame_idx in video_segments:
                positions = {}
                for obj_id, mask in video_segments[frame_idx].items():
                    box = mask_to_box(mask)
                    if box:
                        x1, y1, x2, y2 = box
                        positions[obj_id] = ((x1 + x2) / 2, y2)
            else:
                positions = {}

            # Generate tactical view
            team_assignments = {
                oid: tracking_info.get(oid, {}).get('team', 0)
                for oid in positions
            }

            tactical = self.tactical_view.render(
                positions,
                frame.shape[:2],
                team_assignments,
                self.team_colors
            )

            # Resize tactical view to output dimensions
            tactical_resized = cv2.resize(tactical, (tactical_width, tactical_height))

            # Add stage label
            self._add_stage_label(tactical_resized, "Stage 5: Tactical 2D View (Homography Transform)")

            # Add legend
            self._add_tactical_legend(tactical_resized, tracking_info, positions)

            writer.write(tactical_resized)

        writer.release()
        return output_path

    def _generate_combined_video(
        self,
        frames: List[np.ndarray],
        video_segments: Dict[int, Dict[int, np.ndarray]],
        tracking_info: Dict[int, Dict],
        output_dir: str,
        fps: float,
        smoothed_positions: List[Dict[int, Tuple[float, float]]] = None,
    ) -> str:
        """Generate combined video with all stages (final output)."""
        output_path = os.path.join(output_dir, "stage6_combined.mp4")
        height, width = frames[0].shape[:2]
        writer = self._get_video_writer(output_path, fps, width, height)

        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            # Build transformer
            self.tactical_view.build_transformer(frame)

            # Get positions
            if smoothed_positions and frame_idx < len(smoothed_positions):
                positions = smoothed_positions[frame_idx]
            elif frame_idx in video_segments:
                positions = {}
                for obj_id, mask in video_segments[frame_idx].items():
                    box = mask_to_box(mask)
                    if box:
                        x1, y1, x2, y2 = box
                        positions[obj_id] = ((x1 + x2) / 2, y2)
            else:
                positions = {}

            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    mask_2d = mask.squeeze()
                    info = tracking_info.get(obj_id, {})
                    team_id = info.get('team', 0)
                    color = self.team_colors.get(team_id, (128, 128, 128))

                    # Draw colored mask
                    mask_colored = np.zeros_like(annotated)
                    mask_colored[mask_2d] = color
                    annotated = cv2.addWeighted(annotated, 1.0, mask_colored, 0.4, 0)

                    # Draw bounding box
                    box = mask_to_box(mask)
                    if box:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                        # Create label
                        jersey_number = info.get('jersey_number')
                        player_name = info.get('player_name')

                        if jersey_number and player_name:
                            label = f"#{jersey_number} {player_name}"
                        elif jersey_number:
                            label = f"#{jersey_number}"
                        else:
                            team_name = info.get('team_name', '')
                            label = f"#{obj_id} {team_name}"

                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Add tactical view overlay
            if positions:
                team_assignments = {
                    oid: tracking_info.get(oid, {}).get('team', 0)
                    for oid in positions
                }
                tactical = self.tactical_view.render(
                    positions,
                    frame.shape[:2],
                    team_assignments,
                    self.team_colors
                )
                annotated = create_combined_view(annotated, tactical)

            # Add stage label
            self._add_stage_label(annotated, "SwishVision: Complete Pipeline")

            writer.write(annotated)

        writer.release()
        return output_path

    def _add_stage_label(self, frame: np.ndarray, text: str):
        """Add stage label to top of frame."""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _add_tactical_legend(
        self,
        frame: np.ndarray,
        tracking_info: Dict[int, Dict],
        positions: Dict[int, Tuple[float, float]]
    ):
        """Add team legend to tactical view."""
        height = frame.shape[0]
        y_offset = height - 60

        for team_id in [0, 1]:
            color = self.team_colors.get(team_id, (128, 128, 128))
            name = self.team_names[team_id] if team_id < len(self.team_names) else f"Team {team_id}"

            # Count players on this team currently visible
            count = sum(1 for oid in positions if tracking_info.get(oid, {}).get('team') == team_id)

            # Draw legend entry
            cv2.circle(frame, (30, y_offset), 10, color, -1)
            cv2.circle(frame, (30, y_offset), 10, (255, 255, 255), 2)
            cv2.putText(frame, f"{name} ({count})", (50, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    def _generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr))
        return colors


def generate_portfolio_from_tracking_result(
    frames: List[np.ndarray],
    video_segments: Dict[int, Dict[int, np.ndarray]],
    tracking_info: Dict[int, Dict],
    output_dir: str,
    fps: float = 30.0,
    team_names: Tuple[str, str] = ("Indiana Pacers", "Oklahoma City Thunder"),
    team_colors: Dict[int, Tuple[int, int, int]] = None,
    smoothed_positions: List[Dict[int, Tuple[float, float]]] = None,
    detections_per_frame: Dict[int, sv.Detections] = None,
) -> Dict[str, str]:
    """
    Convenience function to generate portfolio videos from tracking results.

    Args:
        frames: List of video frames
        video_segments: Tracking segments from PlayerTracker
        tracking_info: Tracking info from PlayerTracker
        output_dir: Output directory
        fps: Video frame rate
        team_names: Team names tuple
        team_colors: Optional custom team colors
        smoothed_positions: Optional smoothed positions
        detections_per_frame: Optional raw detections (for stage 1)

    Returns:
        Dict mapping stage name to video path
    """
    # Default colors based on team names
    if team_colors is None:
        team_colors = {
            0: TEAM_COLORS.get(team_names[0], (0, 255, 0)),
            1: TEAM_COLORS.get(team_names[1], (0, 0, 255)),
            -1: (0, 255, 255),  # Referees
        }

    # If no detections provided, create empty dict
    if detections_per_frame is None:
        detections_per_frame = {}

    generator = PortfolioGenerator(
        team_colors=team_colors,
        team_names=team_names,
    )

    return generator.generate_stage_videos(
        frames=frames,
        video_segments=video_segments,
        tracking_info=tracking_info,
        detections_per_frame=detections_per_frame,
        output_dir=output_dir,
        fps=fps,
        smoothed_positions=smoothed_positions,
    )
