# SwishVision

Offline computer-vision pipeline that turns one broadcast-angle basketball video into persistent player tracking, team labels, jersey-number OCR, and a 2-D tactical mini-court.

**Source of truth:** [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) v1.0.0. When code and spec disagree, the spec wins and the code is updated.

---

## Layout

```
swishvision/
├── docs/                      Specification, traceability, architecture, changelog
├── src/swishvision/           Single Python package (importable as `swishvision`)
│   ├── cli.py                 Single CLI entry point: `python -m swishvision …`
│   ├── config.py              All tunable values (the "macros everywhere" rule)
│   ├── pipeline/              Detection, segmentation, teams, jersey, court, tactical, smoothing
│   ├── render/                Stage-video + portfolio rendering
│   ├── runners/               Per-mode entry points called by the CLI
│   └── data/                  Static lookup tables (rosters, brand colours)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── acceptance/            (v1.1 — automates AC-01..AC-10 from the spec)
├── scripts/
│   └── slurm/                 sbatch templates for VT ARC (Tinkercliffs)
├── assets/                    (gitignored) ML weights + test videos
├── outputs/                   (gitignored) every run lands here under outputs/<run_id>
└── attic/                     Quarantined v0 code (legacy runners, half-built API/frontend)
```

## Install (Mac / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Then drop a `.env` at the project root with `ROBOFLOW_API_KEY=...` and put weights into `assets/`:

```
assets/
├── checkpoints/sam2.1_hiera_large.pt
├── models/rf-detr-base.pth
└── test_videos/test_game.mp4
```

(All gitignored — transfer via `scp`/`rsync` to any new environment.)

## Run

```bash
# Fast smoke test — 5 random frames × 6 cumulative-stage JPGs (no MP4s):
python -m swishvision smoke

# Full pipeline — six annotated MP4s:
python -m swishvision full

# Re-render team-coloured frames from an existing run's checkpoints:
python -m swishvision rerender --run outputs/run_<id>

# Pre-fetch model weights (run once on a machine with internet):
python -m swishvision warm-cache
```

All runs land in `outputs/<run_id>/`.

## Cluster (VT Tinkercliffs)

```bash
ssh barakaeli@tinkercliffs2.arc.vt.edu
cd /home/$USER/swishvision
git pull origin main
sbatch scripts/slurm/run_full.sbatch
squeue -u $USER
tail -f outputs/slurm-<jobid>.out
```

See `docs/SPECIFICATION.md` §6.5 (NFR-50/NFR-51) and the project memory entry `reference_cluster_tinkercliffs` for partition + allocation details.

## Project status

v1.0.0. See `docs/SPECIFICATION.md` §15 for the v1.1 roadmap and §12 for known issues.
