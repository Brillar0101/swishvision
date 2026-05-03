# Changelog

All notable changes to SwishVision. Versions follow [Semantic Versioning](https://semver.org/).
Spec versions track this changelog: a behaviour change bumps the spec patch version.

## [Unreleased]

### Added
- **Project restructure** — single Python package at `src/swishvision/` (installable via `pip install -e .`).
- **Single CLI** — `python -m swishvision {full,smoke,rerender,clear-checkpoints,warm-cache}` replaces four legacy scripts.
- **`config.py`** — every tunable value in one module (NFR-30 / "macros everywhere").
- **`docs/CHANGELOG.md`**, **`README.md`** rewrite, **`pyproject.toml`** with pinned dev/api extras.
- **`scripts/slurm/`** — sbatch templates for full + smoke runs.
- **`attic/`** — quarantined the half-built FastAPI scaffold (`api/`, `core/`, `models/`, `schemas/`), the unused frontend, and the four legacy runner scripts.

### Changed
- All ML modules moved from `backend/app/ml/` to `src/swishvision/{pipeline,render,data}/`.
- Tests moved to `tests/{unit,integration}/`. Acceptance tests slot under `tests/acceptance/`.
- ML weights and test videos moved to `assets/{checkpoints,models,test_videos}/` (all gitignored).
- Run outputs now standardise on `outputs/<run_id>/` instead of seven `portfolio_outputs_*/` siblings.
- `.gitignore` rewritten to reflect new layout.

### Notes
- The `.venv/` shebangs are invalidated by the directory move; recreate with `python3 -m venv .venv && pip install -e .`.

## [1.0.0] — 2026-05-02

### Added
- `docs/SPECIFICATION.md` v1.0.0 — functional + non-functional requirements, system architecture, pipeline data contracts, acceptance criteria AC-01..AC-10, known issues R-01..R-08, v1.1 roadmap, audited parameter appendix.
- `docs/REQUIREMENTS_TRACEABILITY.md` — maps every requirement to its implementing module and verifying test, with status (✅/🟡/🔴).

### Fixed
- **SAM2 dtype clash on H100/H200/A100.** Both `build_sam2_video_predictor` and `build_sam2_camera_predictor` now have `predictor.model.float()` called immediately after construction, and the outer `torch.amp.autocast(...)` wrapper has been dropped. Avoids the `mat1/mat2 dtype mismatch` (BFloat16 vs Float) error inside `memory_attention`.
- `TeamClassifierWrapper.get_lighter_cluster` now warns instead of silently returning `0` when the underlying classifier lacks the method (still falls back to `0` — full fix in v1.1, see R-01).
- `TeamClassifier.__getstate__` excludes `_training_crops` / `_training_labels` from pickle so checkpoint files don't balloon with raw training crops.
- `player_tracker.py` warn-once guard parser bug (`if frame_idx == keyframe_indices[1] if … else 15:` → properly parenthesised).

### Changed
- `swap_teams` is now a documented operator escape hatch for FR-21 misclassifications.

## [0.x] — pre-spec

The pre-spec history (commits before `27b956e`) is the integration of RF-DETR + ByteTrack + SAM2 + SigLIP + SmolVLM2 into a single Python pipeline. See `git log --until 2026-05-02` for context. No formal versioning prior to v1.0.0.
