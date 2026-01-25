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
from app.ml.ui_config import (
    Colors, TextSize, VideoConfig,
    put_text_pil, add_title_bar, create_player_label, add_stats_overlay
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Video generation
DISTINCT_COLORS_COUNT = 20  # Number of distinct colors for object IDs
BOX_ANNOTATOR_THICKNESS = 2
MASK_OVERLAY_ALPHA = 0.5  # Segmentation stage
TEAM_OVERLAY_ALPHA = 0.4  # Team/jersey stages
BOX_THICKNESS = 3  # Bounding box thickness for teams/jersey
CONTOUR_THICKNESS = 2  # Mask contour thickness

# Tactical view dimensions (16:9 aspect ratio)
TACTICAL_VIDEO_WIDTH = 1280
TACTICAL_VIDEO_HEIGHT = 720

# Label positioning offsets
LABEL_OFFSET_Y = 35  # Offset above box for ID labels
LABEL_OFFSET_Y_SMALL = 40  # Offset for title labels

# Color generation
HUE_MAX = 180  # Maximum hue value for HSV


def mask_to_box(mask):
    """Convert binary mask to bounding box."""
    mask_2d = mask.squeeze()
    ys, xs = np.where(mask_2d)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = TEAM_OVERLAY_ALPHA
) -> np.ndarray:
    """
    Draw a colored mask overlay on a frame.

    Args:
        frame: Frame to draw on
        mask: Binary mask
        color: BGR color tuple
        alpha: Transparency (0.0-1.0)

    Returns:
        Frame with mask overlay
    """
    mask_2d = mask.squeeze()
    mask_colored = np.zeros_like(frame)
    mask_colored[mask_2d] = color
    return cv2.addWeighted(frame, 1.0, mask_colored, alpha, 0)


def draw_player_annotation(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    label: str,
    alpha: float = TEAM_OVERLAY_ALPHA,
    draw_box: bool = True
) -> np.ndarray:
    """
    Draw complete player annotation (mask, box, label).

    Args:
        frame: Frame to annotate
        mask: Player mask
        color: BGR color tuple
        label: Text label
        alpha: Mask transparency
        draw_box: Whether to draw bounding box

    Returns:
        Annotated frame
    """
    # Draw mask overlay
    annotated = draw_mask_overlay(frame, mask, color, alpha)

    # Draw bounding box and label
    box = mask_to_box(mask)
    if box and draw_box:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        annotated = create_player_label(annotated, label, (x1, y1, x2, y2), color, use_pil=True)

    return annotated


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
        stages_to_generate: List[int] = None,
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
            stages_to_generate: List of stage numbers to generate (e.g., [1, 2]). None = all stages.

        Returns:
            Dict mapping stage name to video path
        """
        os.makedirs(output_dir, exist_ok=True)

        height, width = frames[0].shape[:2]
        frame_count = len(frames)

        # Default: generate all stages
        if stages_to_generate is None:
            stages_to_generate = [1, 2, 3, 4, 5, 6]

        output_paths = {}

        # Stage 1: Raw Detection
        if 1 in stages_to_generate:
            print("Generating Stage 1: Raw Detection...")
            output_paths['detection'] = self._generate_detection_video(
                frames, detections_per_frame, output_dir, fps
            )

        # Stage 2: Segmentation
        if 2 in stages_to_generate:
            print("Generating Stage 2: Segmentation...")
            output_paths['segmentation'] = self._generate_segmentation_video(
                frames, video_segments, output_dir, fps
            )

        # Stage 3: Team Classification
        if 3 in stages_to_generate:
            print("Generating Stage 3: Team Classification...")
            output_paths['teams'] = self._generate_teams_video(
                frames, video_segments, tracking_info, output_dir, fps
            )

        # Stage 4: Jersey Detection
        if 4 in stages_to_generate:
            print("Generating Stage 4: Jersey Detection...")
            output_paths['jersey'] = self._generate_jersey_video(
                frames, video_segments, tracking_info, output_dir, fps
            )

        # Stage 5: Tactical View Only
        if 5 in stages_to_generate:
            print("Generating Stage 5: Tactical View...")
            output_paths['tactical'] = self._generate_tactical_video(
                frames, video_segments, tracking_info, output_dir, fps, smoothed_positions
            )

        # Stage 6: Combined View
        if 6 in stages_to_generate:
            print("Generating Stage 6: Combined View...")
            output_paths['combined'] = self._generate_combined_video(
                frames, video_segments, tracking_info, output_dir, fps, smoothed_positions
            )

        print(f"Generated {len(output_paths)} portfolio videos")
        return output_paths

    def _get_video_writer(self, path: str, fps: float, width: int, height: int):
        """Create high-quality video writer."""
        return VideoConfig.create_writer(path, fps, width, height, use_h264=True)

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
        box_annotator = sv.BoxAnnotator(thickness=BOX_ANNOTATOR_THICKNESS)

        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            # Add stage label
            self._add_stage_label(annotated, "Stage 1: Player Detection (RF-DETR)")

            if frame_idx in detections_per_frame:
                detections = detections_per_frame[frame_idx]
                annotated = box_annotator.annotate(annotated, detections)

                # Add detection count with professional styling
                stats = {"Detected Players": len(detections)}
                annotated = add_stats_overlay(annotated, stats, position='bottom-left', use_pil=True)

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
        colors = self._generate_distinct_colors(DISTINCT_COLORS_COUNT)

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
                    annotated = cv2.addWeighted(annotated, 1.0, mask_colored, MASK_OVERLAY_ALPHA, 0)

                    # Draw outline
                    contours, _ = cv2.findContours(
                        mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annotated, contours, -1, color, CONTOUR_THICKNESS)

                    # Add object ID with professional label
                    box = mask_to_box(mask)
                    if box:
                        x1, y1 = int(box[0]), int(box[1])
                        annotated = put_text_pil(annotated, f"ID:{obj_id}", (x1, y1 - LABEL_OFFSET_Y),
                                                font_size=TextSize.LABEL, color=Colors.WHITE,
                                                bg_color=color, padding=6)

                # Add segment count with stats overlay
                stats = {"Player Segments": len(video_segments[frame_idx])}
                annotated = add_stats_overlay(annotated, stats, position='bottom-left', use_pil=True)

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
                    info = tracking_info.get(obj_id, {})
                    team_id = info.get('team', 0)
                    team_name = info.get('team_name', 'Unknown')
                    color = self.team_colors.get(team_id, (128, 128, 128))

                    team_counts[team_id] = team_counts.get(team_id, 0) + 1

                    # Draw player annotation (mask + box + label)
                    annotated = draw_player_annotation(annotated, mask, color, team_name)

                # Add team counts with stats overlay
                stats = {}
                for team_id, count in team_counts.items():
                    if count > 0:
                        if team_id == -1:
                            name = "Referees"
                        elif team_id < len(self.team_names):
                            name = self.team_names[team_id]
                        else:
                            name = f"Team {team_id}"
                        stats[name] = count
                annotated = add_stats_overlay(annotated, stats, position='bottom-left', use_pil=True)

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
                    info = tracking_info.get(obj_id, {})
                    team_id = info.get('team', 0)
                    color = self.team_colors.get(team_id, (128, 128, 128))

                    jersey_number = info.get('jersey_number')
                    player_name = info.get('player_name')

                    if jersey_number:
                        identified_count += 1

                    # Create label with jersey number and player name
                    if jersey_number and player_name:
                        label = f"#{jersey_number} {player_name}"
                    elif jersey_number:
                        label = f"#{jersey_number}"
                    else:
                        label = f"ID:{obj_id}"

                    # Draw player annotation (mask + box + label)
                    annotated = draw_player_annotation(annotated, mask, color, label)

                # Add identification stats with professional overlay
                total = len(video_segments[frame_idx])
                stats = {
                    "Identified Players": f"{identified_count}/{total}",
                    "Success Rate": f"{int(100 * identified_count / total) if total > 0 else 0}%"
                }
                annotated = add_stats_overlay(annotated, stats, position='bottom-left', use_pil=True)

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

        writer = self._get_video_writer(output_path, fps, TACTICAL_VIDEO_WIDTH, TACTICAL_VIDEO_HEIGHT)

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
            tactical_resized = cv2.resize(tactical, (TACTICAL_VIDEO_WIDTH, TACTICAL_VIDEO_HEIGHT))

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
                    info = tracking_info.get(obj_id, {})
                    team_id = info.get('team', 0)
                    color = self.team_colors.get(team_id, (128, 128, 128))

                    # Create professional label
                    jersey_number = info.get('jersey_number')
                    player_name = info.get('player_name')

                    if jersey_number and player_name:
                        label = f"#{jersey_number} {player_name}"
                    elif jersey_number:
                        label = f"#{jersey_number}"
                    else:
                        team_name = info.get('team_name', '')
                        label = f"#{obj_id} {team_name}"

                    # Draw player annotation (mask + box + label)
                    annotated = draw_player_annotation(annotated, mask, color, label)

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
        """Add professional stage label to top of frame."""
        add_title_bar(frame, text, height=70, bg_color=Colors.OVERLAY_DARK, text_color=Colors.WHITE, use_pil=True)

    def _add_tactical_legend(
        self,
        frame: np.ndarray,
        tracking_info: Dict[int, Dict],
        positions: Dict[int, Tuple[float, float]]
    ):
        """Add professional team legend to tactical view."""
        # Create stats dictionary for overlay
        stats = {}
        for team_id in [0, 1]:
            name = self.team_names[team_id] if team_id < len(self.team_names) else f"Team {team_id}"
            count = sum(1 for oid in positions if tracking_info.get(oid, {}).get('team') == team_id)
            stats[name] = count

        # Add professional stats overlay
        add_stats_overlay(frame, stats, position='bottom-left', use_pil=True)

    def _generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors."""
        colors = []
        for i in range(n):
            hue = int(HUE_MAX * i / n)
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
