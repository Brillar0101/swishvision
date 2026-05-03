"""Smoke runner: emit 5 random frames × 6 cumulative-stage JPGs.

Replaces the legacy ``run_pipeline_5frames.py`` script in ``attic/``. Used as a
cluster-side smoke test before kicking off a full ``run_pipeline_streaming_all``
run.
"""
from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from swishvision.config import (
    RENDER_TEAM_OVERLAY_ALPHA,
    RENDER_TITLE_BAR_HEIGHT,
    SAMPLE_EDGE_TRIM,
    SAMPLE_SEED,
    SMOKE_NUM_RANDOM_FRAMES,
)
from swishvision.pipeline.tactical import create_combined_view
from swishvision.pipeline.tracker import PlayerTracker, mask_to_box
from swishvision.render.ui import Colors, add_title_bar


log = logging.getLogger(__name__)
JPEG_Q = 92


def _add_stage_label(frame: np.ndarray, text: str) -> None:
    add_title_bar(
        frame,
        text,
        height=RENDER_TITLE_BAR_HEIGHT,
        bg_color=Colors.OVERLAY_DARK,
        text_color=Colors.WHITE,
        use_pil=True,
    )


def _distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    out = []
    for i in range(max(n, 1)):
        hue = int(180 * i / max(n, 1))
        bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        out.append(tuple(int(c) for c in bgr))
    return out


def _draw_box(frame: np.ndarray, box, color, label: str) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    if not label:
        return
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _overlay_mask(frame: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    mask_2d = mask.squeeze().astype(bool)
    if not mask_2d.any():
        return frame
    overlay = np.zeros_like(frame)
    overlay[mask_2d] = color
    blended = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)
    contours, _ = cv2.findContours(
        mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, color, 2)
    return blended


def _build_label(obj_id: int, info: Dict, level: str) -> str:
    if level == "id":
        return f"ID:{obj_id}"
    if level == "team":
        team_name = info.get("team_name") or ""
        return team_name or f"#{obj_id}"
    if level == "jersey":
        jersey = info.get("jersey_number")
        player = info.get("player_name")
        if jersey and player:
            return f"#{jersey} {player}"
        if jersey:
            return f"#{jersey}"
        return info.get("team_name") or f"#{obj_id}"
    return ""


def render_stacked_stages(
    frame: np.ndarray,
    frame_idx: int,
    sample_idx: int,
    masks: Dict[int, np.ndarray],
    tracking_info: Dict[int, Dict],
    smoothed: Dict[int, Tuple[float, float]],
    team_classifier,
    tactical_view,
    out_dir: Path,
) -> List[Path]:
    """Save 6 cumulative-stage JPGs for one source frame."""
    paths: List[Path] = []
    base = frame.copy()

    boxes: Dict[int, list] = {}
    for obj_id, mask in masks.items():
        b = mask_to_box(mask)
        if b is not None:
            boxes[obj_id] = b

    seg_palette = _distinct_colors(max(len(masks), 1))
    seg_color = {oid: seg_palette[i % len(seg_palette)] for i, oid in enumerate(masks.keys())}

    def _save(stage_num: int, name: str, image: np.ndarray) -> None:
        _add_stage_label(image, f"Stage {stage_num} — {name}  [#{sample_idx + 1} f={frame_idx}]")
        path = out_dir / f"stacked_{sample_idx + 1:02d}_f{frame_idx:04d}_{stage_num}_{name.lower().replace(' + ', '_').replace(' ', '_')}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        paths.append(path)

    # Stage 1: raw
    _save(1, "Raw Frame", base.copy())

    # Stage 2: + detection
    s2 = base.copy()
    for obj_id, box in boxes.items():
        _draw_box(s2, box, (0, 255, 0), f"ID:{obj_id}")
    _save(2, "+ RF-DETR Detection", s2)

    # Stage 3: + segmentation (per-object distinct colour)
    s3 = base.copy()
    for obj_id, mask in masks.items():
        s3 = _overlay_mask(s3, mask, seg_color[obj_id], RENDER_TEAM_OVERLAY_ALPHA)
    for obj_id, box in boxes.items():
        _draw_box(s3, box, seg_color[obj_id], f"ID:{obj_id}")
    _save(3, "+ SAM2 Segmentation", s3)

    # Stage 4: + team colours
    s4 = base.copy()
    for obj_id, mask in masks.items():
        info = tracking_info.get(obj_id, {})
        color = team_classifier.get_team_color(info.get("team", 0))
        s4 = _overlay_mask(s4, mask, color, RENDER_TEAM_OVERLAY_ALPHA)
    for obj_id, box in boxes.items():
        info = tracking_info.get(obj_id, {})
        color = team_classifier.get_team_color(info.get("team", 0))
        _draw_box(s4, box, color, _build_label(obj_id, info, "team"))
    _save(4, "+ Team Classification", s4)

    # Stage 5: + jersey labels
    s5 = base.copy()
    for obj_id, mask in masks.items():
        info = tracking_info.get(obj_id, {})
        color = team_classifier.get_team_color(info.get("team", 0))
        s5 = _overlay_mask(s5, mask, color, RENDER_TEAM_OVERLAY_ALPHA)
    for obj_id, box in boxes.items():
        info = tracking_info.get(obj_id, {})
        color = team_classifier.get_team_color(info.get("team", 0))
        _draw_box(s5, box, color, _build_label(obj_id, info, "jersey"))
    _save(5, "+ Jersey OCR (SmolVLM2)", s5)

    # Stage 6: + tactical mini-court
    s6 = s5.copy()
    if smoothed and tactical_view._last_transformer is not None:
        team_assignments = {oid: tracking_info.get(oid, {}).get("team", 0) for oid in smoothed}
        tc = {
            0: team_classifier.get_team_color(0),
            1: team_classifier.get_team_color(1),
            -1: (0, 255, 255),
        }
        tactical = tactical_view.render(smoothed, frame.shape[:2], team_assignments, tc)
        s6 = create_combined_view(s6, tactical)
    _save(6, "+ Tactical 2D View", s6)

    return paths


def _pick_random_indices(num_frames: int, num_samples: int) -> List[int]:
    rng = random.Random(SAMPLE_SEED)
    lo = int(num_frames * SAMPLE_EDGE_TRIM)
    hi = int(num_frames * (1 - SAMPLE_EDGE_TRIM))
    return sorted(rng.sample(range(lo, hi), num_samples))


def run(
    *,
    video: Path,
    run_dir: Path,
    team_names: Tuple[str, str],
    swap_teams: bool,
    max_seconds: float | None,
    num_random_frames: int = SMOKE_NUM_RANDOM_FRAMES,
    sam2_checkpoint: Path,
    sam2_config: str,
) -> int:
    frame_dir = run_dir / "stacked_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Stale .frames_cache from earlier runs makes SAM2 batch propagate the wrong
    # frame count -- always clear it before a fresh run.
    cache_dir = run_dir / ".frames_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        log.info("cleared stale frame cache at %s", cache_dir)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    full_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    capped_n = min(full_n, int(max_seconds * fps)) if max_seconds else full_n
    log.info("video=%s frames=%s fps=%.2f processing=%s", video, full_n, fps, capped_n)

    indices = _pick_random_indices(capped_n, num_random_frames)
    log.info("random sample indices: %s", indices)

    # Monkey-patch the orchestrator to use OUR indices and skip MP4 generation.
    original_portfolio = PlayerTracker._generate_portfolio_frames
    original_videos = PlayerTracker._generate_stage_videos

    def patched_portfolio(self, frames, video_segments, tracking_info, team_classifier,
                          sample_indices, output_dir, tactical_view, smoothed_positions):
        usable = [i for i in indices if i < len(frames)]
        log.info("rendering stacked stages for %s", usable)
        all_paths: List[str] = []
        for s_idx, f_idx in enumerate(usable):
            paths = render_stacked_stages(
                frames[f_idx], f_idx, s_idx,
                video_segments.get(f_idx, {}),
                tracking_info,
                smoothed_positions[f_idx]
                if smoothed_positions is not None and f_idx < len(smoothed_positions)
                else {},
                team_classifier, tactical_view, frame_dir,
            )
            all_paths.extend(str(p) for p in paths)
            log.info("frame %s: wrote %s stacked stages", f_idx, len(paths))
        return all_paths

    def skip_videos(*_a, **_kw):
        log.info("MP4 generation skipped (smoke mode)")
        return {}

    PlayerTracker._generate_portfolio_frames = patched_portfolio
    PlayerTracker._generate_stage_videos = skip_videos
    try:
        tracker = PlayerTracker(
            sam2_checkpoint=str(sam2_checkpoint),
            sam2_config=sam2_config,
        )
        tracker.process_video_with_tracking(
            video_path=str(video),
            output_dir=str(run_dir),
            team_names=team_names,
            use_bytetrack=True,
            use_sam2_segmentation=True,
            use_streaming_sam2=True,
            max_total_objects=20,
            use_court_mask_filter=True,
            max_seconds=max_seconds,
            resume=False,
            swap_teams=swap_teams,
            num_sample_frames=num_random_frames,
        )
    finally:
        PlayerTracker._generate_portfolio_frames = original_portfolio
        PlayerTracker._generate_stage_videos = original_videos

    log.info("=== Smoke Run Done ===")
    log.info("frames written to: %s", frame_dir)
    return 0
