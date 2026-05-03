"""Run the SwishVision pipeline against a short clip and emit, for each of 5
random frames, six cumulative-stage JPGs.

Stage progression (each stage stacks on top of the previous overlays):
  1 raw            -- original frame, header only
  2 +detection     -- per-object bounding boxes (track IDs)
  3 +segmentation  -- per-object SAM2 masks (distinct colours)
  4 +teams         -- recolour masks/boxes by team, label team name
  5 +jersey        -- replace track-ID labels with #jersey + player name
  6 +tactical      -- composite the 2D mini-court onto frame 5

To keep the run tractable on Apple-Silicon MPS we use the small SAM2 weights
and cap the clip length. Heavy video writing is monkey-patched away.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

# Load Roboflow API key etc. before importing modules that initialise models.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
if not os.getenv("ROBOFLOW_API_KEY"):
    raise SystemExit("ROBOFLOW_API_KEY not set; expected in backend/.env")

from swishvision.pipeline.tracker import PlayerTracker, mask_to_box
from swishvision.pipeline.tactical import create_combined_view
from swishvision.render.ui import Colors, add_title_bar


SEED = 42
NUM_RANDOM_FRAMES = 5
MAX_SECONDS = 8.0
EDGE_TRIM = 0.10
FRAME_DIR = os.path.join(os.path.dirname(__file__), "..", "local_outputs", "random_frames")
JPEG_Q = 92


def _add_stage_label(frame: np.ndarray, text: str) -> None:
    add_title_bar(
        frame,
        text,
        height=70,
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


def _overlay_mask(frame: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
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
    """Pick the right label text per stage level."""
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


def _render_stacked_stages(
    frame: np.ndarray,
    frame_idx: int,
    sample_idx: int,
    masks: Dict[int, np.ndarray],
    tracking_info: Dict[int, Dict],
    smoothed: Dict[int, Tuple[float, float]],
    team_classifier,
    tactical_view,
    out_dir: str,
) -> List[str]:
    """Save 6 cumulative-stage JPGs for one source frame."""
    paths: List[str] = []
    base = frame.copy()

    # ---- Stage 1: raw ----
    s1 = base.copy()
    _add_stage_label(s1, f"Stage 1 — Raw Frame  [#{sample_idx + 1} f={frame_idx}]")
    p = os.path.join(out_dir, f"stacked_{sample_idx + 1:02d}_f{frame_idx:04d}_1_raw.jpg")
    cv2.imwrite(p, s1, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    paths.append(p)

    # Pre-compute boxes once
    boxes: Dict[int, list] = {}
    for obj_id, mask in masks.items():
        b = mask_to_box(mask)
        if b is not None:
            boxes[obj_id] = b

    seg_palette = _distinct_colors(max(len(masks), 1))
    seg_color = {oid: seg_palette[i % len(seg_palette)] for i, oid in enumerate(masks.keys())}

    # ---- Stage 2: raw + detection (boxes) ----
    s2 = base.copy()
    for obj_id, box in boxes.items():
        _draw_box(s2, box, (0, 255, 0), f"ID:{obj_id}")
    _add_stage_label(s2, f"Stage 2 — + RF-DETR Detection  [#{sample_idx + 1} f={frame_idx}]")
    p = os.path.join(out_dir, f"stacked_{sample_idx + 1:02d}_f{frame_idx:04d}_2_detection.jpg")
    cv2.imwrite(p, s2, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    paths.append(p)

    # ---- Stage 3: + segmentation (per-object distinct mask colour) ----
    s3 = base.copy()
    for obj_id, mask in masks.items():
        s3 = _overlay_mask(s3, mask, seg_color[obj_id], alpha=0.45)
    for obj_id, box in boxes.items():
        _draw_box(s3, box, seg_color[obj_id], f"ID:{obj_id}")
    _add_stage_label(s3, f"Stage 3 — + SAM2 Segmentation  [#{sample_idx + 1} f={frame_idx}]")
    p = os.path.join(out_dir, f"stacked_{sample_idx + 1:02d}_f{frame_idx:04d}_3_segmentation.jpg")
    cv2.imwrite(p, s3, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    paths.append(p)

    # ---- Stage 4: + team classification (recolour by team) ----
    s4 = base.copy()
    for obj_id, mask in masks.items():
        info = tracking_info.get(obj_id, {})
        team_id = info.get("team", 0)
        color = team_classifier.get_team_color(team_id)
        s4 = _overlay_mask(s4, mask, color, alpha=0.45)
    for obj_id, box in boxes.items():
        info = tracking_info.get(obj_id, {})
        team_id = info.get("team", 0)
        color = team_classifier.get_team_color(team_id)
        _draw_box(s4, box, color, _build_label(obj_id, info, "team"))
    _add_stage_label(s4, f"Stage 4 — + Team Classification  [#{sample_idx + 1} f={frame_idx}]")
    p = os.path.join(out_dir, f"stacked_{sample_idx + 1:02d}_f{frame_idx:04d}_4_teams.jpg")
    cv2.imwrite(p, s4, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    paths.append(p)

    # ---- Stage 5: + jersey OCR (swap labels) ----
    s5 = base.copy()
    for obj_id, mask in masks.items():
        info = tracking_info.get(obj_id, {})
        team_id = info.get("team", 0)
        color = team_classifier.get_team_color(team_id)
        s5 = _overlay_mask(s5, mask, color, alpha=0.45)
    for obj_id, box in boxes.items():
        info = tracking_info.get(obj_id, {})
        team_id = info.get("team", 0)
        color = team_classifier.get_team_color(team_id)
        _draw_box(s5, box, color, _build_label(obj_id, info, "jersey"))
    _add_stage_label(s5, f"Stage 5 — + Jersey OCR (SmolVLM2)  [#{sample_idx + 1} f={frame_idx}]")
    p = os.path.join(out_dir, f"stacked_{sample_idx + 1:02d}_f{frame_idx:04d}_5_jersey.jpg")
    cv2.imwrite(p, s5, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    paths.append(p)

    # ---- Stage 6: + tactical mini-court overlay ----
    s6 = s5.copy()  # build directly on stage 5 so it's truly cumulative
    if smoothed and tactical_view._last_transformer is not None:
        team_assignments = {oid: tracking_info.get(oid, {}).get("team", 0) for oid in smoothed}
        tc = {
            0: team_classifier.get_team_color(0),
            1: team_classifier.get_team_color(1),
            -1: (0, 255, 255),
        }
        tactical = tactical_view.render(smoothed, frame.shape[:2], team_assignments, tc)
        s6 = create_combined_view(s6, tactical)
    _add_stage_label(s6, f"Stage 6 — + Tactical 2D View  [#{sample_idx + 1} f={frame_idx}]")
    p = os.path.join(out_dir, f"stacked_{sample_idx + 1:02d}_f{frame_idx:04d}_6_tactical.jpg")
    cv2.imwrite(p, s6, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    paths.append(p)

    return paths


def _install_patches(random_indices: List[int]):
    """Patch PlayerTracker so the run produces only stacked-stage JPGs."""
    def patched_portfolio(
        self,
        frames,
        video_segments,
        tracking_info,
        team_classifier,
        sample_indices,
        output_dir,
        tactical_view,
        smoothed_positions,
    ):
        os.makedirs(FRAME_DIR, exist_ok=True)
        # Use OUR random indices, but drop any beyond the actually-processed clip.
        usable = [i for i in random_indices if i < len(frames)]
        if not usable:
            print(f"[5frames] WARNING: none of the random indices {random_indices} fit "
                  f"in the {len(frames)}-frame clip; falling back to evenly spaced")
            usable = sample_indices
        print(f"[5frames] rendering stacked stages for indices: {usable}")

        all_paths: List[str] = []
        for s_idx, f_idx in enumerate(usable):
            frame = frames[f_idx]
            masks = video_segments.get(f_idx, {})
            smoothed = (
                smoothed_positions[f_idx]
                if smoothed_positions is not None and f_idx < len(smoothed_positions)
                else {}
            )
            paths = _render_stacked_stages(
                frame, f_idx, s_idx, masks, tracking_info, smoothed,
                team_classifier, tactical_view, FRAME_DIR,
            )
            all_paths.extend(paths)
            print(f"[5frames] frame {f_idx}: wrote {len(paths)} stacked stages "
                  f"(masks={len(masks)})")
        return all_paths

    def skip_videos(*_args, **_kwargs):
        print("[5frames] _generate_stage_videos skipped (videos disabled)")
        return {}

    PlayerTracker._generate_portfolio_frames = patched_portfolio
    PlayerTracker._generate_stage_videos = skip_videos


def _pick_random_indices(num_frames: int) -> List[int]:
    rng = random.Random(SEED)
    lo = int(num_frames * EDGE_TRIM)
    hi = int(num_frames * (1 - EDGE_TRIM))
    indices = sorted(rng.sample(range(lo, hi), NUM_RANDOM_FRAMES))
    print(f"[5frames] {num_frames}-frame clip; selected: {indices}")
    return indices


def main() -> None:
    video_path = os.path.join(os.path.dirname(__file__), "..", "test_videos", "test_game.mp4")
    output_dir = os.path.join(os.path.dirname(__file__), "portfolio_outputs_5frames")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    # Stale .frames_cache from a previous (longer) run will cause SAM2's
    # init_state to propagate across every cached PNG -- not just the clip
    # we extract this run. Clear it so SAM2 only sees the current MAX_SECONDS clip.
    import shutil
    cache_dir = os.path.join(output_dir, ".frames_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[5frames] cleared stale frame cache at {cache_dir}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    full_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    capped_n = min(full_n, int(MAX_SECONDS * fps))
    print(f"[5frames] full video {full_n} frames @ {fps:.2f} fps; "
          f"processing first {capped_n} frames ({MAX_SECONDS}s)")

    indices = _pick_random_indices(capped_n)
    _install_patches(indices)

    # Use SAM2-small to keep MPS runtime reasonable.
    tracker = PlayerTracker(
        sam2_checkpoint="checkpoints/sam2.1_hiera_small.pt",
        sam2_config="sam2.1_hiera_s",
    )
    result = tracker.process_video_with_tracking(
        video_path=video_path,
        output_dir=output_dir,
        team_names=("Indiana Pacers", "Oklahoma City Thunder"),
        # Streaming SAM2 path requires a camera predictor we don't have locally;
        # the bytetrack+SAM2 dispatch has no batch fallback. Use SAM2-only batch.
        use_bytetrack=False,
        use_sam2_segmentation=True,
        use_streaming_sam2=False,
        max_total_objects=15,
        use_court_mask_filter=True,
        max_seconds=MAX_SECONDS,
        resume=False,
        num_sample_frames=NUM_RANDOM_FRAMES,
    )

    print("\n=== Done ===")
    print(f"Players tracked: {result.get('players_tracked')}")
    print(f"Stacked stage frames in: {FRAME_DIR}")


if __name__ == "__main__":
    main()
