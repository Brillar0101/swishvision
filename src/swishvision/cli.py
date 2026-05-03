"""Single CLI entry point for SwishVision.

Replaces the four legacy scripts (``run_pipeline_streaming_all.py``,
``run_pipeline_5frames.py``, ``rerender_swapped.py``, ``debug_pipeline.py``)
that have been quarantined to ``attic/``.

Usage:

    python -m swishvision full   --video assets/test_videos/test_game.mp4
    python -m swishvision smoke  --video assets/test_videos/test_game.mp4
    python -m swishvision rerender --run outputs/run_2026_05_02
    python -m swishvision clear-checkpoints --run outputs/run_2026_05_02
    python -m swishvision warm-cache

Subcommands map directly onto the spec's stage flow (``docs/SPECIFICATION.md``).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from swishvision import __version__
from swishvision.config import (
    DEFAULT_MAX_TOTAL_OBJECTS,
    DEFAULT_NUM_SAMPLE_FRAMES,
    DEFAULT_SAM2_CONFIG_LARGE,
    DEFAULT_SAM2_LARGE_CHECKPOINT,
    DEFAULT_TEST_VIDEO,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    SMOKE_MAX_SECONDS,
    SMOKE_NUM_RANDOM_FRAMES,
)


log = logging.getLogger(__name__)


def _common_setup(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Roboflow / HF clients read env vars at import time -- load before importing.
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _resolve_run_dir(name: str | None) -> Path:
    if name is None:
        from datetime import datetime

        name = "run_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    p = (OUTPUTS_DIR / name).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_full(args: argparse.Namespace) -> int:
    """Run the full pipeline end-to-end and emit six stage MP4s."""
    _common_setup(args.verbose)

    from swishvision.pipeline.tracker import PlayerTracker

    run_dir = _resolve_run_dir(args.run)
    log.info("Pipeline starting: video=%s run_dir=%s", args.video, run_dir)

    tracker = PlayerTracker(
        sam2_checkpoint=str(args.sam2_checkpoint),
        sam2_config=args.sam2_config,
    )
    result = tracker.process_video_with_tracking(
        video_path=str(args.video),
        output_dir=str(run_dir),
        team_names=tuple(args.team_names),
        use_bytetrack=args.use_bytetrack,
        use_sam2_segmentation=True,
        use_streaming_sam2=args.streaming_sam2,
        max_total_objects=args.max_objects,
        use_court_mask_filter=True,
        max_seconds=args.max_seconds,
        resume=args.resume,
        swap_teams=args.swap_teams,
        num_sample_frames=DEFAULT_NUM_SAMPLE_FRAMES,
    )

    log.info("=== Pipeline Complete ===")
    log.info("Players tracked: %s", result.get("players_tracked"))
    log.info("Stage videos: %s", result.get("stage_videos"))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Smoke-test runner: 5 random frames × 6 cumulative-stage JPGs, no MP4s."""
    _common_setup(args.verbose)

    from swishvision.runners import smoke as smoke_runner  # type: ignore[attr-defined]

    return smoke_runner.run(
        video=args.video,
        run_dir=_resolve_run_dir(args.run),
        team_names=tuple(args.team_names),
        swap_teams=args.swap_teams,
        max_seconds=args.max_seconds,
        num_random_frames=args.num_frames,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
    )


def cmd_rerender(args: argparse.Namespace) -> int:
    """Re-render stacked-stage JPGs from an existing run's checkpoints."""
    _common_setup(args.verbose)

    from swishvision.runners import rerender as rerender_runner  # type: ignore[attr-defined]

    return rerender_runner.run(
        run_dir=Path(args.run).resolve(),
        cluster_to_team=dict(zip([0, 1], args.team_names)),
        video=args.video,
        max_seconds=args.max_seconds,
        num_random_frames=args.num_frames,
    )


def cmd_clear_checkpoints(args: argparse.Namespace) -> int:
    """Wipe checkpoint state for a run while leaving its outputs alone."""
    _common_setup(args.verbose)

    import shutil

    run_dir = Path(args.run).resolve()
    for sub in (".checkpoints", ".frames_cache"):
        target = run_dir / sub
        if target.exists():
            shutil.rmtree(target)
            log.info("removed %s", target)
    return 0


def cmd_warm_cache(args: argparse.Namespace) -> int:
    """Pre-fetch Roboflow + HF model weights to local cache (login-node helper)."""
    _common_setup(args.verbose)

    from swishvision.pipeline.detection import PlayerRefereeDetector
    from swishvision.pipeline.jersey import JerseyDetector

    PlayerRefereeDetector()
    JerseyDetector()
    log.info("model caches warm")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swishvision",
        description="SwishVision pipeline CLI — see docs/SPECIFICATION.md",
    )
    p.add_argument("--version", action="version", version=f"swishvision {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_video = lambda sp: sp.add_argument(  # noqa: E731
        "--video", type=Path, default=DEFAULT_TEST_VIDEO,
        help="input MP4 (default: assets/test_videos/test_game.mp4)",
    )

    # full
    sp_full = sub.add_parser("full", help="run full pipeline; emit six stage MP4s")
    common_video(sp_full)
    sp_full.add_argument("--run", default=None, help="run-dir name under outputs/")
    sp_full.add_argument("--max-seconds", type=float, default=None,
                         help="cap clip length in seconds")
    sp_full.add_argument("--max-objects", type=int, default=DEFAULT_MAX_TOTAL_OBJECTS)
    sp_full.add_argument("--team-names", nargs=2,
                         default=["Indiana Pacers", "Oklahoma City Thunder"],
                         metavar=("LIGHTER", "DARKER"))
    sp_full.add_argument("--swap-teams", action="store_true",
                         help="invert auto-detected lighter/darker mapping")
    sp_full.add_argument("--use-bytetrack", action=argparse.BooleanOptionalAction,
                         default=True)
    sp_full.add_argument("--streaming-sam2", action=argparse.BooleanOptionalAction,
                         default=True)
    sp_full.add_argument("--sam2-checkpoint", type=Path,
                         default=DEFAULT_SAM2_LARGE_CHECKPOINT)
    sp_full.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG_LARGE)
    sp_full.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    sp_full.set_defaults(func=cmd_full)

    # smoke
    sp_smoke = sub.add_parser(
        "smoke",
        help="5 random frames × 6 cumulative-stage JPGs (no MP4 output)",
    )
    common_video(sp_smoke)
    sp_smoke.add_argument("--run", default=None)
    sp_smoke.add_argument("--max-seconds", type=float, default=SMOKE_MAX_SECONDS)
    sp_smoke.add_argument("--num-frames", type=int, default=SMOKE_NUM_RANDOM_FRAMES)
    sp_smoke.add_argument("--team-names", nargs=2,
                          default=["Indiana Pacers", "Oklahoma City Thunder"])
    sp_smoke.add_argument("--swap-teams", action="store_true")
    sp_smoke.add_argument("--sam2-checkpoint", type=Path,
                          default=DEFAULT_SAM2_LARGE_CHECKPOINT)
    sp_smoke.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG_LARGE)
    sp_smoke.set_defaults(func=cmd_smoke)

    # rerender
    sp_re = sub.add_parser("rerender",
                           help="re-render stacked frames from an existing run")
    sp_re.add_argument("--run", required=True, help="run-dir under outputs/ to reuse")
    common_video(sp_re)
    sp_re.add_argument("--max-seconds", type=float, default=None)
    sp_re.add_argument("--num-frames", type=int, default=SMOKE_NUM_RANDOM_FRAMES)
    sp_re.add_argument("--team-names", nargs=2,
                       default=["Oklahoma City Thunder", "Indiana Pacers"],
                       metavar=("CLUSTER0", "CLUSTER1"))
    sp_re.set_defaults(func=cmd_rerender)

    # clear-checkpoints
    sp_cc = sub.add_parser("clear-checkpoints",
                           help="wipe .checkpoints and .frames_cache for a run")
    sp_cc.add_argument("--run", required=True)
    sp_cc.set_defaults(func=cmd_clear_checkpoints)

    # warm-cache
    sp_wc = sub.add_parser("warm-cache",
                           help="pre-fetch Roboflow / HF weights (run on login node)")
    sp_wc.set_defaults(func=cmd_warm_cache)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
