# Requirements Traceability — SwishVision v1.0.0

Maps every requirement in [`SPECIFICATION.md`](SPECIFICATION.md) to the code that implements it and the test that verifies it. Audit performed 2026-05-02 against `main`.

| Status | Meaning |
| --- | --- |
| ✅ | Implemented and unit/integration-tested. |
| 🟡 | Implemented; test missing or insufficient. v1.1 must add. |
| 🔴 | Not implemented. |

## Functional requirements

| Req | Status | Implemented in | Tested by |
| --- | --- | --- | --- |
| FR-01 Accept MP4 | ✅ | `PlayerTracker._extract_frames` | `tests/test_player_tracking.py` (smoke) |
| FR-02 `max_seconds` cap | ✅ | same | manual; 🟡 add unit test |
| FR-03 Frame cache | 🟡 | same; runner clears cache only in `run_pipeline_5frames.py` | none — see R-06 |
| FR-10 Detect everyone | ✅ | `PlayerRefereeDetector.detect` | `tests/test_player_tracking.py` |
| FR-11 Court-mask filter | 🟡 | `PlayerTracker._filter_by_court_mask` | none |
| FR-12 Persistent tracker IDs | 🟡 | `PlayerRefereeDetector.detect_and_track` | `tests/test_player_tracking.py` (partial) |
| FR-13 `max_total_objects` | 🟡 | `PlayerTracker.process_video_with_tracking` | none |
| FR-14 Two SAM2 dispatch modes | 🟡 | `player_tracker.py:1326` (streaming), `:1538` (batch) | manual; integration test stubbed |
| FR-15 SAM2 fp32 forced | ✅ | `predictor.model.float()` after each `build_sam2_*_predictor` | none — 🟡 add regression test |
| FR-16 Mask cleaning | ✅ | `filter_segments_by_distance` calls | none |
| FR-20 Train classifier | ✅ | `TeamClassifier.fit` | none — 🟡 add unit test |
| FR-21 Auto-detect lighter cluster | 🔴 | `TeamClassifier.get_lighter_cluster` exists; **wrapper falls back to 0** when underlying is `sports.TeamClassifier` (R-01) | none |
| FR-22 Map cluster→team | ✅ | `PlayerTracker.process_video_with_tracking` (team_names + swap_teams) | none |
| FR-23 Predict per-player team | ✅ | `TeamClassifierWrapper.predict_team` | none |
| FR-30 Jersey-number boxes | ✅ | `JerseyDetector.detect` | `tests/test_jersey_detection.py` |
| FR-31 OCR | ✅ | `JerseyDetector.recognize_number` | none — 🟡 add unit test with synthetic crop |
| FR-32 IoS match | ✅ | `JerseyDetector` matching logic | `tests/test_jersey_detection.py` |
| FR-33 Consecutive validation | ✅ | `ConsecutiveValueTracker` | `tests/test_jersey_detection.py` |
| FR-34 Roster lookup | ✅ | `team_rosters.get_player_name` | none |
| FR-40 Court keypoints | ✅ | `CourtDetector` | `tests/test_homography.py` |
| FR-41 Homography fit | ✅ | `TacticalView.build_transformer` | `tests/test_homography.py` |
| FR-42 Project positions | ✅ | `TacticalView` | `tests/test_tactical_pipeline.py` |
| FR-43 Path smoothing | ✅ | `path_smoothing.py` | none — 🟡 add unit test |
| FR-44 Render mini-court | ✅ | `tactical_view.py:render` | `tests/test_tactical_pipeline.py` |
| FR-50 Six stage MP4s | ✅ | `portfolio_generator.PortfolioGenerator` | none — 🟡 add filesize/duration check |
| FR-51 Sampled JPGs | ✅ | `PlayerTracker._generate_portfolio_frames` | `tests/test_portfolio_helpers.py` |
| FR-52 Configurable colours | ✅ | `team_rosters.TEAM_COLORS` + `rerender_swapped.TeamClassifierShim` | none |
| FR-53 Stage label banner | ✅ | `_add_stage_label` in runner + `portfolio_generator` | none |
| FR-60 Per-stage checkpoint | ✅ | `PipelineCheckpoint` | none — 🟡 add unit test |
| FR-61 Resume | ✅ | `process_video_with_tracking` resume branches | none — 🟡 add integration test |
| FR-62 Pickle `__getstate__` | ✅ | `TeamClassifier.__getstate__` | none — 🟡 add unit test |
| FR-63 SLURM-friendly | ✅ | `backend/.env`, `run_5frames.slurm`, `run_h200.slurm` | manual |

## Non-functional requirements

| Req | Status | Notes |
| --- | --- | --- |
| NFR-01 A100 throughput ≤ 10 min | 🟡 | Demonstrated 2026-05-02 once; not pinned by automated benchmark. |
| NFR-02 H200 throughput ≤ 6 min | 🟡 | One run failed mid-flight on dtype before fp32 fix; rerun pending. |
| NFR-03 GPU memory ≤ 60 GB / 80 GB | 🟡 | Not measured. |
| NFR-04 Disk footprint ≤ 10 GB / 1 GB | 🟡 | Single observed run was ~8.4 GB intermediate. Unverified for full clip. |
| NFR-10 No silent failures | ✅ | Verified by code review (`R-01` is the one tracked exception, logged once). |
| NFR-11 Deterministic team K-means | ✅ | `KMEANS_RANDOM_STATE = 42`, `KMEANS_N_INIT = 10`. |
| NFR-12 Deterministic random frames | ✅ | `SEED = 42` in `run_pipeline_5frames.py`. |
| NFR-20 Structured logs | 🔴 | Currently `print()`. Migrate to `logging` in v1.1 (R-04). |
| NFR-21 Visible checkpoint state | ✅ | `<output_dir>/.checkpoints/*.pkl`. |
| NFR-30 Module-level constants | 🟡 | Mostly compliant; ByteTrack args + jersey OCR interval not yet hoisted. |
| NFR-31 Function length | 🔴 | `process_video_with_tracking` ~780 LOC. v1.1 split. |
| NFR-32 File length | 🔴 | `player_tracker.py` (1860), `portfolio_generator.py` (638). v1.1 split. |
| NFR-33 Type hints | 🟡 | Public APIs done; some private helpers untyped. |
| NFR-40 No secrets in source | ✅ | `git ls-files \| grep -E '\.env\|\.pth\|\.pt'` returns empty. |
| NFR-41 Per-user IDE excluded | ✅ | `.claude/settings.local.json` in `.gitignore`. |
| NFR-50 Linux + CUDA | ✅ | Tinkercliffs verified. |
| NFR-51 Compute-node-friendly | 🟡 | No automated pre-warm; operator runs it manually. v1.1 add a warmup script. |

## Acceptance criteria

| AC | Status | How to verify |
| --- | --- | --- |
| AC-01 Clean exit | 🟡 | grep `slurm-*.out` for `=== Pipeline Complete ===`. |
| AC-02 Six MP4s, ≥ 80 % duration | 🟡 | `ffprobe` script (write in v1.1). |
| AC-03 ≥ 10 player IDs | 🟡 | `len(tracking_info) >= 10` post-run; one observed run = 15. |
| AC-04 ≥ 80 % team-named | 🟡 | aggregate count post-run; not yet measured. |
| AC-05 ≥ 6 unique jerseys | 🟡 | post-run count; not yet measured. |
| AC-06 Court success ≥ 90 % frames | 🟡 | not measured. |
| AC-07 Operator confirms team direction | 🟡 | manual eyeball on Stage 4 JPG. |
| AC-08 Wall-clock ≤ NFR thresholds | 🟡 | timestamp diff in slurm log. |
| AC-09 No fatal warnings | 🟡 | grep slurm log. |
| AC-10 Portfolio colours match intent | 🟡 | manual. Verified once on 2026-05-02 with rerender_swapped overrides. |

## Test inventory

| File | Lines | Status | What it covers |
| --- | --- | --- | --- |
| `tests/test_constants.py` | small | 🟡 | Sanity bounds. |
| `tests/test_homography.py` | medium | ✅ | Court keypoint → homography. |
| `tests/test_integration.py` | stub | 🔴 | Empty / placeholder; populate in v1.1. |
| `tests/test_jersey_detection.py` | medium | ✅ | OCR + IoS + ConsecutiveValueTracker. |
| `tests/test_player_tracking.py` | medium | 🟡 | Detection smoke; no tracker-persistence test. |
| `tests/test_portfolio_helpers.py` | small | 🟡 | Drawing helpers. |
| `tests/test_tactical_pipeline.py` | medium | 🟡 | Render produces non-empty frame. |

## v1.1 work derived from this matrix

In priority order:

1. Populate `tests/test_integration.py` so a full short-clip run is one `pytest -k integration` away.
2. Add `tests/test_acceptance.py` automating AC-01..AC-10.
3. Lift `get_lighter_cluster` into `TeamClassifierWrapper` (FR-21 → ✅).
4. Migrate `print()` → `logging` (NFR-20 → ✅).
5. Hoist remaining inline numbers to module constants (NFR-30 cleanup).
6. Split `process_video_with_tracking` into per-stage methods (NFR-31, NFR-32).
7. CI: GitHub Actions running `pytest backend/tests` on push.
