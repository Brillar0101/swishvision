"""Rerender runner: re-emit stacked-stage JPGs from an existing run's checkpoints.

Use this when the team→cluster mapping was wrong on the first run -- swaps the
cluster→team-name lookup and re-renders the 30 JPGs without any GPU work.
"""
from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Dict, Tuple

import cv2

from swishvision.config import (
    SAMPLE_EDGE_TRIM,
    SAMPLE_SEED,
    SMOKE_NUM_RANDOM_FRAMES,
)
from swishvision.data.team_rosters import TEAM_COLORS, TEAM_ROSTERS, get_player_name
from swishvision.pipeline.tactical import TacticalView
from swishvision.runners.smoke import render_stacked_stages


log = logging.getLogger(__name__)

# High-contrast BGR overrides (per project preference). See memory entry
# ``project_team_color_gotcha``.
TEAM_COLOR_OVERRIDES_BGR: Dict[str, Tuple[int, int, int]] = {
    "Oklahoma City Thunder": (255, 0, 0),    # pure blue
    "Indiana Pacers":        (0, 255, 255),  # pure yellow
}
REFEREE_COLOR_BGR: Tuple[int, int, int] = (0, 165, 255)  # orange


def _hex_to_bgr(value):
    if isinstance(value, tuple):
        return value
    h = value.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


class TeamClassifierShim:
    """Minimal stand-in matching the attributes the renderer reads."""

    def __init__(self, cluster_to_team: Dict[int, str]):
        self.team_names = dict(cluster_to_team)
        self.team_colors = {
            cluster: TEAM_COLOR_OVERRIDES_BGR.get(name, _hex_to_bgr(TEAM_COLORS.get(name, "#7F7F7F")))
            for cluster, name in cluster_to_team.items()
        }
        self.team_colors[-1] = REFEREE_COLOR_BGR

    def get_team_color(self, t):
        return self.team_colors.get(t, (128, 128, 128))

    def get_team_name(self, t):
        return self.team_names.get(t, f"Team {t}")


def run(
    *,
    run_dir: Path,
    cluster_to_team: Dict[int, str],
    video: Path,
    max_seconds: float | None,
    num_random_frames: int = SMOKE_NUM_RANDOM_FRAMES,
) -> int:
    ckpt = run_dir / ".checkpoints"
    if not ckpt.exists():
        raise FileNotFoundError(f"no .checkpoints under {run_dir}")

    with open(ckpt / "tracking_info_with_jerseys.pkl", "rb") as f:
        tracking_info = pickle.load(f)
    with open(ckpt / "video_segments.pkl", "rb") as f:
        video_segments = pickle.load(f)
    with open(ckpt / "smoothed_positions.pkl", "rb") as f:
        smoothed_positions = pickle.load(f)

    # Re-label every tracked object using the corrected cluster→team mapping.
    # Keep cluster id unchanged -- only the human-readable label flips.
    for info in tracking_info.values():
        t = info.get("team")
        if t in cluster_to_team:
            new_name = cluster_to_team[t]
            info["team_name"] = new_name
            jersey = info.get("jersey_number")
            info.pop("player_name", None)
            if jersey and new_name in TEAM_ROSTERS:
                pn = get_player_name(new_name, jersey)
                if pn:
                    info["player_name"] = pn

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap_n = int(max_seconds * fps) if max_seconds else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while len(frames) < cap_n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    log.info("loaded %s frames", len(frames))

    tv = TacticalView()
    tv.build_transformer(frames[0])

    rng = random.Random(SAMPLE_SEED)
    lo = int(len(frames) * SAMPLE_EDGE_TRIM)
    hi = int(len(frames) * (1 - SAMPLE_EDGE_TRIM))
    indices = sorted(rng.sample(range(lo, hi), num_random_frames))
    log.info("re-rendering: %s", indices)

    classifier = TeamClassifierShim(cluster_to_team)
    out_dir = run_dir / "stacked_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    for s_idx, f_idx in enumerate(indices):
        masks = video_segments.get(f_idx, {})
        smoothed = (
            smoothed_positions[f_idx] if f_idx < len(smoothed_positions) else {}
        )
        paths = render_stacked_stages(
            frames[f_idx], f_idx, s_idx, masks, tracking_info, smoothed,
            classifier, tv, out_dir,
        )
        log.info("frame %s: wrote %s JPGs", f_idx, len(paths))

    log.info("=== Rerender Done ===")
    return 0
