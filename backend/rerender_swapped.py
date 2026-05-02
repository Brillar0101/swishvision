"""Re-render the 30 stacked-stage JPGs with team labels swapped.

Uses checkpoint data from a previous successful pipeline run -- no SAM2 or
jersey OCR re-run needed. Use this when the pipeline assigned cluster <-> team
in the wrong direction and you want to flip without reprocessing.

The fix: keep tracking_info[obj_id]['team'] (the K-means cluster id) UNCHANGED.
Only the *name* mapped to each cluster needs to flip, and the
TeamClassifier-shaped object the renderer asks for colors must map each cluster
to its true jersey colour.
"""
from __future__ import annotations

import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

import cv2

from app.ml.tactical_view import TacticalView
from app.ml.team_rosters import TEAM_COLORS, TEAM_ROSTERS, get_player_name
from run_pipeline_5frames import (
    EDGE_TRIM,
    FRAME_DIR,
    MAX_SECONDS,
    NUM_RANDOM_FRAMES,
    SEED,
    _render_stacked_stages,
)


CKPT = "portfolio_outputs_5frames/.checkpoints"

# After running once, the tracking_info had each cluster assigned to the wrong
# real-world team. Flip the human-readable label per cluster, but leave the
# cluster id (the int K-means produced) alone -- otherwise the renderer's color
# lookup ends up double-flipping and nothing changes.
#
# Cluster 0 in this run is the WHITE-jersey cluster (Thunder), cluster 1 is
# yellow (Pacers). Adjust here if a future run lands the opposite way.
CLUSTER_TO_TEAM = {
    0: "Oklahoma City Thunder",
    1: "Indiana Pacers",
}

# Override the rosters' brand hex codes with high-contrast pure blue/yellow
# per project preference. Referees recoloured to orange so they don't clash
# with Pacers yellow.
TEAM_COLOR_OVERRIDES_BGR = {
    "Oklahoma City Thunder": (255, 0, 0),    # pure blue
    "Indiana Pacers":        (0, 255, 255),  # pure yellow
}
REFEREE_COLOR_BGR = (0, 165, 255)            # orange


class TeamClassifierShim:
    """Minimal stand-in matching the attributes the renderer reads."""

    def __init__(self, cluster_to_team):
        self.team_names = dict(cluster_to_team)
        self.team_colors = {
            cluster: TEAM_COLOR_OVERRIDES_BGR.get(name, (128, 128, 128))
            for cluster, name in cluster_to_team.items()
        }
        self.team_colors[-1] = REFEREE_COLOR_BGR

    def get_team_color(self, t):
        return self.team_colors.get(t, (128, 128, 128))

    def get_team_name(self, t):
        return self.team_names.get(t, f"Team {t}")


def main() -> None:
    with open(f"{CKPT}/tracking_info_with_jerseys.pkl", "rb") as f:
        tracking_info = pickle.load(f)
    with open(f"{CKPT}/video_segments.pkl", "rb") as f:
        video_segments = pickle.load(f)
    with open(f"{CKPT}/smoothed_positions.pkl", "rb") as f:
        smoothed_positions = pickle.load(f)

    # Re-label each tracked object using the corrected cluster->team mapping.
    # Do NOT rewrite the cluster id; that would double-flip with the colour map.
    for info in tracking_info.values():
        t = info.get("team")
        if t in CLUSTER_TO_TEAM:
            new_name = CLUSTER_TO_TEAM[t]
            info["team_name"] = new_name
            jersey = info.get("jersey_number")
            info.pop("player_name", None)
            if jersey and new_name in TEAM_ROSTERS:
                pn = get_player_name(new_name, jersey)
                if pn:
                    info["player_name"] = pn

    cap = cv2.VideoCapture("../test_videos/test_game.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap_n = (
        int(MAX_SECONDS * fps)
        if MAX_SECONDS
        else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    frames = []
    while len(frames) < cap_n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames")

    tv = TacticalView()
    tv.build_transformer(frames[0])

    rng = random.Random(SEED)
    lo = int(len(frames) * EDGE_TRIM)
    hi = int(len(frames) * (1 - EDGE_TRIM))
    indices = sorted(rng.sample(range(lo, hi), NUM_RANDOM_FRAMES))
    print(f"Re-rendering: {indices}")

    classifier_shim = TeamClassifierShim(CLUSTER_TO_TEAM)
    os.makedirs(FRAME_DIR, exist_ok=True)

    for s_idx, f_idx in enumerate(indices):
        masks = video_segments.get(f_idx, {})
        smoothed = (
            smoothed_positions[f_idx]
            if f_idx < len(smoothed_positions)
            else {}
        )
        paths = _render_stacked_stages(
            frames[f_idx],
            f_idx,
            s_idx,
            masks,
            tracking_info,
            smoothed,
            classifier_shim,
            tv,
            FRAME_DIR,
        )
        print(f"frame {f_idx}: wrote {len(paths)} JPGs")

    print("=== Re-render Done ===")


if __name__ == "__main__":
    main()
