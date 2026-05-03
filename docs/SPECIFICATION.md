# SwishVision — Project Specification

| Field | Value |
| --- | --- |
| Document version | **1.0.0** |
| Status | Draft for review |
| Date | 2026-05-02 |
| Project ID | ARC project 7569 (Tinkercliffs allocation `cjones_swish`) |
| PI | Creed Jones (`crjones4@vt.edu`) |
| Author | Barakaeli Lawuo (`barakaeli@vt.edu`) |
| Department | Electrical & Computer Engineering, Virginia Tech |
| Repository | `Brillar0101/swishvision` (default branch `main`) |
| Related docs | `backend/ARCHITECTURE.md`, `Context.txt` (legacy) |

This document is the source of truth for *what SwishVision is supposed to do*. The code in `backend/` is the implementation against this specification. When code and spec disagree, the spec wins and the code is updated; when both have to change together, both are updated in the same commit.

---

## 1. Executive summary

SwishVision is an offline computer-vision pipeline that turns a single broadcast-quality basketball video into a labelled, analysable representation of the game. Given an MP4 of a game (or game segment), SwishVision produces:

1. Persistent IDs for every player and referee on the court, frame by frame.
2. Pixel-level segmentation masks for each tracked person.
3. Team membership per player (two-team unsupervised clustering).
4. Jersey numbers per player, validated across multiple frames.
5. Player names, looked up against a known roster.
6. A 2D tactical "mini-court" with smoothed player trajectories projected via homography.
7. Six annotated MP4s (one per pipeline stage) and a set of sampled annotated frames suitable for portfolio review.

The system is intended for **post-game analysis at college basketball programs** (target customer per project description). It is **not** real-time and is **not** a refereeing system. Its role is to lower the cost of generating per-player metrics that today require a film-room human.

Version 1.0.0 of this spec describes the system as it exists in the `main` branch on the date above, with the gaps and known issues called out explicitly in §13–§15.

---

## 2. Goals and non-goals

### 2.1 In-scope goals

- **G1.** Process a single broadcast-angle basketball video end-to-end and emit annotated outputs without per-game manual labelling.
- **G2.** Maintain stable player IDs across short occlusions (≥ 20 seconds at 30 fps).
- **G3.** Recover team identity automatically (without the operator hand-labelling team A vs team B per game).
- **G4.** Recognise jersey numbers reliably enough to attach NBA roster names where the team is in the configured roster set.
- **G5.** Produce a 2D mini-court projection that a coach can read at a glance.
- **G6.** Be re-runnable from intermediate state when a stage fails or is interrupted (checkpoint/resume).
- **G7.** Run on Virginia Tech ARC GPU resources (Tinkercliffs A100 / H200) without code changes between dev and prod.

### 2.2 Explicit non-goals (v1)

- **NG1.** Real-time / live streaming. The pipeline is batch.
- **NG2.** Multi-camera fusion. Single broadcast feed only.
- **NG3.** Officiating decisions, foul detection, possession state, shot tracking. (Tracked separately as future work — see §15.)
- **NG4.** Custom model training inside the pipeline. Models are pretrained / fine-tuned offline; the pipeline consumes weights.
- **NG5.** A user-facing web product. The frontend in `frontend/` is unfinished and not in scope for v1.

### 2.3 Stakeholders

| Stakeholder | Interest |
| --- | --- |
| Creed Jones (PI) | Project oversight, research direction, evaluation. |
| Barakaeli Lawuo (developer) | Build, run, evaluate the system. |
| Future college basketball staff (target user) | Receive annotated outputs and per-player metrics. |
| ARC operations | Resource allocation, job behaviour on shared cluster. |

---

## 3. Glossary

| Term | Meaning |
| --- | --- |
| **Tracker ID** | Integer assigned to a single physical person across the video. Persistent across frames; survives short occlusions. |
| **Cluster ID** | K-means cluster (0 or 1) assigned to a player based on jersey appearance. Arbitrary; mapped to a real team name in a separate step. |
| **Keyframe** | Frame on which a fresh detection pass is run (vs. a frame on which we only propagate masks from the previous keyframe). |
| **Tactical view** | 2D top-down mini-court rendered from homography of the broadcast frame. |
| **Stage** | One of the six annotated outputs the pipeline emits. |
| **Portfolio** | Static JPGs / videos generated specifically for showing pipeline behaviour, separate from any per-game analysis. |

---

## 4. System overview

### 4.1 Black-box view

```
                 ┌─────────────────────────────┐
   game.mp4 ───▶ │      SwishVision pipeline   │ ───▶ annotated MP4 ×6
                 │                             │ ───▶ tracking_info.pkl
                 │                             │ ───▶ video_segments (chunked)
                 │                             │ ───▶ smoothed_positions.pkl
                 │                             │ ───▶ portfolio JPGs
                 └─────────────────────────────┘
```

### 4.2 Internal pipeline (white-box view)

The pipeline is seven stages. Each stage reads from disk-backed checkpoint state, processes, writes back, and marks itself complete. Resume = skip stages whose completion marker exists.

```
[1] Frame extraction
       └─→ frames as JPGs in .frames_cache/
[2] Player crop collection (1 fps sampling, court-mask filtered)
       └─→ crops.pkl
[3] Team classifier training (SigLIP embeddings + K-means k=2)
       └─→ team_classifier.pkl
[4] Detection + tracking
       (a) ByteTrack pre-pass with RF-DETR per-frame, OR
       (b) SAM2-only mode: keyframe detection + mask propagation
       └─→ bytetrack_detections.pkl, video_segments(_chunk_*).pkl
[5] Team assignment
       └─→ tracking_info_with_teams.pkl
[6] Jersey OCR (every 5th frame, RF-DETR + SmolVLM2, validated by ConsecutiveValueTracker n=3)
       └─→ tracking_info_with_jerseys.pkl, jersey_numbers.pkl
[7] Court detection + homography + path smoothing
       └─→ smoothed_positions.pkl
[render] 6 stage MP4s + sampled portfolio frames
```

### 4.3 Module map

| Module | Responsibility | LOC |
| --- | --- | --- |
| `app/ml/player_referee_detector.py` | RF-DETR detection + ByteTrack pre-pass. | 230 |
| `app/ml/player_tracker.py` | Pipeline orchestrator. Stage dispatch, SAM2 integration, checkpointing. | 1860 |
| `app/ml/team_classifier.py` | SigLIP embedding + K-means. | 318 |
| `app/ml/jersey_detector.py` | RF-DETR jersey-number detector + SmolVLM2 OCR + validation. | 305 |
| `app/ml/court_detector.py` | Court keypoint detection. | 199 |
| `app/ml/tactical_view.py` | Homography fitting + 2D mini-court rendering. | 430 |
| `app/ml/path_smoothing.py` | Trajectory cleaning (jump removal, Savitzky-Golay smoothing). | 281 |
| `app/ml/portfolio_generator.py` | Six-stage MP4 rendering. | 638 |
| `app/ml/team_rosters.py` | NBA roster lookup + colour map. | 82 |
| `app/ml/ui_config.py` | Drawing helpers, colour palette, fonts. | 452 |

Pipeline entry points (consolidate to one in v1.1, see §16):

| Script | Purpose |
| --- | --- |
| `backend/run_pipeline_streaming_all.py` | Production: full video, six MP4s, ByteTrack + streaming SAM2. |
| `backend/run_pipeline_5frames.py` | Cluster smoke-test: produces 5 random frames × 6 cumulative-overlay JPGs. |
| `backend/rerender_swapped.py` | Post-processing: re-render frames from existing checkpoint with corrected team mapping. |
| `backend/debug_pipeline.py` | Developer-only debugging. |

---

## 5. Functional requirements

Each requirement has an ID, a one-sentence statement, an acceptance test, and a pointer to the implementing module(s). Implementation pointers may be a function or file, current as of `main`.

### 5.1 Input handling

- **FR-01 — Accept a single MP4 input.** The pipeline accepts a path to an `.mp4` file, opens it via OpenCV, and rejects with a `ValueError` if it cannot be opened. *Verifies: opens valid file, fails on missing/corrupt.* Implementation: `PlayerTracker._extract_frames` ([player_tracker.py:392](../backend/app/ml/player_tracker.py#L392)).
- **FR-02 — Honour `max_seconds` cap.** When `max_seconds` is non-null, the pipeline processes only the first `max_seconds × fps` frames. Implementation: same.
- **FR-03 — Cache extracted frames to disk.** Frames are written to `<output_dir>/.frames_cache/NNNNN.jpg` and reused on resume. Stale caches from earlier runs that contain a different number of frames must be cleared before the run, since SAM2 reads frame count from the directory. *Known v1 gap: `run_pipeline_5frames.py` clears the cache; `run_pipeline_streaming_all.py` does not. v1.1 fix.*

### 5.2 Detection & tracking

- **FR-10 — Detect players, referees, ball, jersey numbers, rim per frame.** RF-DETR fine-tuned model (`basketball-player-detection-3-ycjdo/4`) returns class IDs in `{0..7}` plus 8 (referee). Implementation: `PlayerRefereeDetector.detect` ([player_referee_detector.py](../backend/app/ml/player_referee_detector.py)).
- **FR-11 — Filter detections to on-court regions.** When `use_court_mask_filter=True`, detections whose centre falls outside the detected court polygon are dropped (removes spectators in the stands). Implementation: `PlayerTracker._filter_by_court_mask`.
- **FR-12 — Persistent tracker IDs.** ByteTrack maintains a tracker ID across frames with `lost_track_buffer=600` (≈ 20 s @ 30 fps), so a player who briefly leaves the frame returns with the same ID. Constants: `track_activation_threshold=0.15`, `minimum_matching_threshold=0.5`, `minimum_consecutive_frames=1`. Implementation: same module.
- **FR-13 — Cap maximum tracked objects.** `max_total_objects` (default 20) limits memory pressure on SAM2. The pipeline keeps the top-N most-frequently-seen tracker IDs.
- **FR-14 — Two SAM2 dispatch modes.** Streaming (`build_sam2_camera_predictor`, lower memory, frame-by-frame) and batch (`build_sam2_video_predictor`, all frames into memory). The dispatch is via `(use_bytetrack, use_sam2_segmentation, use_streaming_sam2, CAMERA_PREDICTOR_AVAILABLE)`. *Tested: A100 batch mode; A100/H200 streaming mode (after fp32 fix in `predictor.model.float()`).*
- **FR-15 — SAM2 weights run in float32.** Both predictor instances are forced to float32 immediately after construction to avoid the bfloat16-vs-float32 dtype clash inside `memory_attention` on H100/H200. Implementation: `predictor.model.float()` calls in `player_tracker.py` after each `build_sam2_*_predictor`.
- **FR-16 — Mask cleaning.** Each SAM2 mask is post-processed by `filter_segments_by_distance(relative_distance=0.03)` to drop disconnected blobs more than 3 % of the frame width from the largest connected component.

### 5.3 Team classification

- **FR-20 — Train a 2-cluster classifier on the input video.** Sample player crops at 1 fps, scale by `0.4` from the centre of each player box, run SigLIP for embeddings, fit K-means with `n_clusters=2`, `random_state=KMEANS_RANDOM_STATE`, `n_init=KMEANS_N_INIT`. Implementation: `team_classifier.py:fit`.
- **FR-21 — Auto-detect the lighter-jersey cluster.** After training, compute the mean HSV of each cluster's crops and select the cluster with the largest `value − saturation` score as "lighter." This determines which K-means cluster id maps to which team name (lighter cluster ↔ first entry of `team_names`). Implementation: `TeamClassifier.get_lighter_cluster` ([team_classifier.py:164](../backend/app/ml/team_classifier.py#L164)). *Known v1 gap: when the wrapper falls back to the third-party `sports.TeamClassifier`, this method is unavailable and the wrapper warns and returns `0`. The operator must use `swap_teams=True` to compensate. v1.1: lift `get_lighter_cluster` into the wrapper itself so it works regardless of underlying implementation.*
- **FR-22 — Map cluster id to real team via configured `team_names`.** The pipeline accepts `team_names=(lighter_team, darker_team)` as input. After cluster assignment, the lighter cluster gets `team_names[0]`, darker gets `team_names[1]`. The `swap_teams=True` flag inverts this mapping for cases where FR-21 is wrong.
- **FR-23 — Predict per-player team membership.** Every tracked object is classified as team `0` or `1` based on its appearance crop. Referees retain team `-1`. Implementation: `TeamClassifierWrapper.predict_team`.

### 5.4 Jersey number recognition

- **FR-30 — Detect jersey number bounding boxes.** Run the jersey RF-DETR model (`basketball-jersey-numbers-ocr/3`) on every 5th frame. Implementation: `JerseyDetector.detect`.
- **FR-31 — OCR each jersey crop.** Pass each detected jersey-number box through SmolVLM2 fine-tuned for jersey-number reading. Output is a string in `0..99` or `00`.
- **FR-32 — Match jersey number to player.** Use Intersection-over-Smaller (IoS) ≥ 0.9 between the jersey box and a player box to assign ownership.
- **FR-33 — Validate OCR with consecutive-value tracker.** Accept a jersey number for a tracker ID only after `n_consecutive=3` independent observations agree. Implementation: `ConsecutiveValueTracker` in `jersey_detector.py`.
- **FR-34 — Look up player name from roster.** When a validated number is matched to a tracker ID with a known team name in `team_rosters.TEAM_ROSTERS`, attach the player's last name to `tracking_info[obj_id]['player_name']`. Implementation: `team_rosters.get_player_name`.

### 5.5 Court & tactical view

- **FR-40 — Detect court keypoints.** Run `basketball-court-detection-2/14` on the first valid frame to produce ≥ 8 keypoints. Implementation: `CourtDetector`.
- **FR-41 — Fit a homography from broadcast frame to 2D court.** `TacticalView.build_transformer` solves the homography. Failure (too few keypoints) is logged and the tactical stage is rendered with an "unavailable" placeholder rather than crashing.
- **FR-42 — Project player positions through the homography.** Each player's anchor (bottom-centre of bounding box) is mapped to court coordinates per frame.
- **FR-43 — Smooth trajectories.** Apply jump detection (`sigma=3.5`, `min_dist=0.6`), short-run removal (`max_jump_run=18`, `pad=2`), linear interpolation, and Savitzky-Golay smoothing (`window=9`, `poly=2`). Implementation: `path_smoothing.py`.
- **FR-44 — Render 2D mini-court.** Players appear as filled circles in their team colour at smoothed positions. Implementation: `tactical_view.py:render`.

### 5.6 Output rendering

- **FR-50 — Emit six stage MP4s.** Stage 1 detection, 2 segmentation, 3 teams, 4 jersey, 5 tactical-only, 6 final-combined. All at the source video's fps and resolution. Implementation: `portfolio_generator.py`.
- **FR-51 — Emit a portfolio of sampled JPGs.** Configurable sample-frame count (default 3 evenly spaced; smoke-test runner uses 5 random); each sample produces six per-stage JPGs. Implementation: `PlayerTracker._generate_portfolio_frames`.
- **FR-52 — Render team colours from a configurable palette.** Brand colours come from `team_rosters.TEAM_COLORS`. Override-friendly so "high-contrast" colour maps (e.g., pure blue for OKC, pure yellow for Pacers, orange for referees) can be applied for slides. Implementation: `rerender_swapped.py:TeamClassifierShim`.
- **FR-53 — Stamp every output with a stage label.** Each frame has a banner `Stage N — <description>` so frames are self-describing in screenshots.

### 5.7 Operations

- **FR-60 — Checkpoint per stage.** After each stage completes, the pipeline writes its output to `<output_dir>/.checkpoints/<stage>.pkl` and updates a `_status.json`-equivalent that lists completed stages. Implementation: `PipelineCheckpoint`.
- **FR-61 — Resume from checkpoint.** When `resume=True`, the pipeline loads completed-stage outputs and skips ahead. When `resume=False`, the pipeline ignores any existing checkpoint and overwrites.
- **FR-62 — Survive SAM2 weight bloat.** `TeamClassifier.__getstate__` excludes `_training_crops` and `_training_labels` from pickle to keep checkpoint size bounded.
- **FR-63 — Run on shared GPU clusters.** The pipeline is submittable as a SLURM job (sbatch) and as an interactive allocation (salloc + tmux for disconnect resilience). Environment variables (e.g., `ROBOFLOW_API_KEY`) are read from `backend/.env`.

---

## 6. Non-functional requirements

### 6.1 Performance

- **NFR-01 — Throughput on A100.** A 25.4-second 1080p clip (760 frames) completes end-to-end in ≤ 10 minutes wall-clock with SAM2 large + jersey OCR + tactical view, on a single A100-80G with 8 CPU cores. *Baseline measured 2026-05-02; subject to model download cache being warm.*
- **NFR-02 — Throughput on H200.** Same workload should complete in ≤ 6 minutes on a single H200-141G.
- **NFR-03 — Memory ceiling.** Peak GPU memory ≤ 60 GB on A100, ≤ 80 GB on H200, including SAM2 large activations across 760 frames.
- **NFR-04 — Disk footprint.** A single completed run produces ≤ 10 GB of intermediate state (checkpoints, `.frames_cache`, video chunks). Outputs (six MP4s + portfolio JPGs) ≤ 1 GB.

### 6.2 Reliability

- **NFR-10 — No silent failures.** Stage errors raise; the only "swallowed" condition allowed is the SAM2 streaming-mid-prompt fallback when the predictor implementation does not support `add_new_prompt_during_track`, and that is logged once per run.
- **NFR-11 — Deterministic team assignment given fixed seed.** With `KMEANS_RANDOM_STATE` fixed and identical training crops, two runs produce identical cluster ids. (`swap_teams` may flip the cluster→name mapping deterministically.)
- **NFR-12 — Deterministic random-frame selection.** The 5-frame smoke test uses `random.seed(SEED)` so the same indices are picked across runs unless the operator changes the seed.

### 6.3 Observability

- **NFR-20 — Structured progress logs.** Each stage prints a banner before starting and an elapsed-time line on completion. tqdm progress bars on long propagation/OCR loops. *Known v1 gap: `print()` is used instead of `logging`. v1.1 migrates to the standard library `logging` module so verbosity is controllable per run.*
- **NFR-21 — Checkpoint progress visible on disk.** Inspecting `<output_dir>/.checkpoints/` shows what stages have completed without running the pipeline.

### 6.4 Maintainability

- **NFR-30 — No magic numbers in module bodies.** Tunable values live in module-level `UPPER_SNAKE_CASE` constants with a comment explaining their origin (typically the Roboflow reference notebook). *In v1: enforced in `path_smoothing.py`; partial in others. v1.1 sweep.*
- **NFR-31 — Function length.** No function exceeds 50 lines; nesting depth ≤ 4. *In v1: violated by `PlayerTracker.process_video_with_tracking` (~780 LOC). v1.1 splits this into per-stage methods.*
- **NFR-32 — File length.** No module exceeds 800 lines. *In v1: `player_tracker.py` (1860 LOC) and `portfolio_generator.py` (638) violate this. v1.1 refactors.*
- **NFR-33 — Type hints on all public signatures.** New code is fully type-annotated. *In v1: largely complied; some private helpers untyped.*

### 6.5 Security & secrets

- **NFR-40 — No secrets in source.** API keys live in `backend/.env`, which is `.gitignore`'d. Verified by `git ls-files | grep -E '\.env|\.pth|\.pt'` returning nothing.
- **NFR-41 — Per-user Claude/IDE settings excluded.** `.claude/settings.local.json` is gitignored.

### 6.6 Portability

- **NFR-50 — Runs on Linux + CUDA.** Tinkercliffs (Rocky Linux + CUDA 13) is the primary target. The runner detects CUDA and falls back to CPU/MPS only for local debugging. Streaming SAM2 requires the camera predictor patch (`backend/install_sam2_camera_predictor.sh`).
- **NFR-51 — Compute-node-friendly.** Roboflow/HF model weights are pre-fetched on the login node before sbatch jobs run, since compute nodes may have restricted outbound networking.

---

## 7. Data contracts

The intermediate state passed between stages is shaped as follows. All keys are stable across versions of this spec.

### 7.1 `tracking_info: Dict[int, Dict]`

Keyed by tracker ID. Per object:

```python
{
    "class": "player" | "referee",
    "confidence": float,           # detection confidence at the prompting frame
    "initial_box": [x1, y1, x2, y2],
    "team": int,                    # 0 or 1 (or -1 for referees)
    "team_name": str,               # human-readable, after team assignment
    "jersey_number": str | None,    # validated, after jersey stage
    "player_name": str | None,      # roster lookup, when team+number both known
}
```

### 7.2 `video_segments: Dict[int, Dict[int, np.ndarray]]`

Keyed by `frame_idx → tracker_id → boolean mask of shape (H, W)`.

For long videos this is chunked: `video_segments_chunk_<i>.pkl` plus a small manifest `video_segments_num_chunks.pkl`.

### 7.3 `smoothed_positions: List[Dict[int, Tuple[float, float]]]`

One entry per frame. Each entry maps tracker ID to a smoothed `(x, y)` in 2D court coordinates.

### 7.4 `bytetrack_detections: Dict[int, sv.Detections]`

Keyed by `frame_idx → supervision.Detections`. Used when `use_bytetrack=True`.

---

## 8. External dependencies

| Dependency | Version constraint | Purpose | Notes |
| --- | --- | --- | --- |
| PyTorch | `>= 2.0` | All ML compute. | CUDA 12.x recommended. |
| `sam2` | upstream + local camera-predictor patch | Segmentation. | Patch via `install_sam2_camera_predictor.sh`. |
| `inference` | `>= 0.9` | Roboflow model loading. | API key in `.env`. |
| `supervision` | `>= 0.16` | ByteTrack + drawing helpers. | |
| `transformers` | bundled with `inference` | SigLIP + SmolVLM2. | |
| `scikit-learn` | `>= 1.3` | K-means. | |
| `scipy` | bundled | Savitzky-Golay smoothing. | |
| `opencv-python` | `>= 4.8` | I/O, drawing. | |
| `roboflow` | `>= 1.1` | Model registry resolution. | |
| `python-dotenv` | `>= 1.0` | Load `.env`. | |
| `tqdm` | `>= 4.65` | Progress bars. | |
| `Pillow` | `>= 10.0` | Image drawing for labels. | |
| FastAPI / SQLAlchemy / Alembic | listed in `requirements.txt` | Future API layer (not used in v1 pipeline). | Out of scope for v1. |

### 8.1 Pretrained model registry

| Model ID (Roboflow Universe) | Used for | Stage |
| --- | --- | --- |
| `basketball-player-detection-3-ycjdo/4` | Player + referee + jersey-number boxes | FR-10, FR-30 |
| `basketball-jersey-numbers-ocr/3` | Jersey-number bounding boxes | FR-30 |
| `basketball-court-detection-2/14` | Court keypoints | FR-40 |
| `facebook/sam2.1-hiera-large` (local checkpoint) | Segmentation | FR-14 |
| SigLIP (HF default checkpoint via `inference`) | Team-jersey embeddings | FR-20 |
| SmolVLM2 (HF, fine-tuned for jersey OCR) | OCR | FR-31 |

### 8.2 Local-only assets (not in git)

- `backend/.env` — Roboflow API key (~38 B).
- `backend/rf-detr-base.pth` — RF-DETR base weights (~355 MB).
- `backend/checkpoints/sam2.1_hiera_large.pt` (~898 MB), `sam2.1_hiera_small.pt` (~184 MB).
- `test_videos/test_game.mp4` — primary regression video (~66 MB).

These must be transferred to any new environment out-of-band (`scp`, `rsync`).

---

## 9. Acceptance criteria

A run is "accepted" against v1.0.0 when **all** of the following are true on the reference clip `test_videos/test_game.mp4`:

| ID | Criterion | Source of truth |
| --- | --- | --- |
| **AC-01** | Pipeline exits cleanly (`=== Pipeline Complete ===`) with no Python exception. | `slurm-*.out` log. |
| **AC-02** | Six stage MP4s exist in the output directory, each ≥ 80 % of the input video's duration. | `ffprobe -v error -show_entries format=duration` per file. |
| **AC-03** | At least 10 distinct `tracker_id`s are present in `tracking_info`. | `len({oid for oid in tracking_info if tracking_info[oid]['class']=='player'})`. |
| **AC-04** | Of those, ≥ 80 % carry a non-null `team_name`. | Aggregate count. |
| **AC-05** | At least 6 distinct validated jersey numbers are present across all tracker IDs. | Count of unique `jersey_number` values. |
| **AC-06** | Court homography succeeded for ≥ 90 % of frames (transformer non-null). | Frame-by-frame check inside tactical render. |
| **AC-07** | The `team_name` assigned to the cluster whose mean HSV value is highest matches the operator's "lighter team" intent (post-`swap_teams` if needed). Operator confirmation only; will be automated in v1.1. | Manual eyeball on portfolio frames. |
| **AC-08** | End-to-end wall-clock ≤ NFR-01 / NFR-02 thresholds. | Timestamp diff in log. |
| **AC-09** | No fatal warnings: zero "MissingEnv", "FileNotFoundError", "RuntimeError". | grep on log. |
| **AC-10** | The 30-JPG portfolio (5 random frames × 6 stages) is generated and team colours visibly match operator intent. | Manual visual review. |

A run that fails one or more AC items is **not** rejected — it is logged with a known-issue tag, and the failure becomes the highest-priority work item against v1.1.

---

## 10. Test strategy

### 10.1 Unit tests (pytest, `backend/tests/`)

| File | Covers |
| --- | --- |
| `test_constants.py` | Constants are within sane bounds (e.g., `KEYFRAME_INTERVAL > 0`, `IOU_THRESHOLD ∈ [0,1]`). |
| `test_homography.py` | Homography fitting succeeds with a known synthetic court. |
| `test_jersey_detection.py` | `ConsecutiveValueTracker` only releases a value after `n_consecutive` agreements. |
| `test_player_tracking.py` | `PlayerRefereeDetector` returns boxes on a sample frame. |
| `test_portfolio_helpers.py` | Drawing helpers produce frames of the expected shape. |
| `test_tactical_pipeline.py` | Smooth → render → composite produces a non-empty image. |
| `test_integration.py` | (Currently a stub — populate in v1.1.) End-to-end smoke test on a 2-second clip. |

### 10.2 Integration test

`run_pipeline_5frames.py` is the cluster-side smoke test. It runs the full inference path on a short clip and emits 30 stacked-stage JPGs. A passing smoke test is a precondition for kicking off the full `run_pipeline_streaming_all.py` on a longer video.

### 10.3 Acceptance test

`backend/tests/test_acceptance.py` (to be added in v1.1) will run `process_video_with_tracking` on `test_videos/test_game.mp4` and assert the AC-01 … AC-10 criteria. Until then, acceptance is operator-driven against §9.

---

## 11. Constraints and assumptions

- **C-01.** Single broadcast camera, fixed-side basketball framing. The court-keypoint model is trained for this.
- **C-02.** Two teams with visually distinct jersey colours. Same-colour matchups (e.g., black vs dark navy) will degrade FR-21 quality.
- **C-03.** Roster lookup is closed-set: teams not in `team_rosters.TEAM_ROSTERS` will never receive `player_name`. Adding a team is a code change.
- **C-04.** Network access to Roboflow/HF Hub is required on first run to fetch model metadata. Subsequent runs use the local cache.
- **C-05.** Tinkercliffs compute nodes may block outbound internet. Pre-warm the cache on the login node before the first sbatch run.

---

## 12. Risks and known issues (carried into v1.1)

| ID | Risk | Mitigation |
| --- | --- | --- |
| **R-01** | `TeamClassifierWrapper.get_lighter_cluster` falls back to `0` when the underlying `sports.TeamClassifier` is in use, defeating auto-detection. | Operator uses `swap_teams=True`. v1.1: lift the HSV check up into the wrapper itself. |
| **R-02** | Multiple pipeline entry-point scripts (`run_pipeline_streaming_all.py`, `run_pipeline_5frames.py`, `debug_pipeline.py`, `rerender_swapped.py`) duplicate orchestration logic. | Consolidate into a single CLI in v1.1 (`python -m swishvision.run --mode {full,smoke,rerender}`). |
| **R-03** | `process_video_with_tracking` is ~780 lines with deeply nested SAM2 dispatch. | Refactor into per-stage methods, dispatch via strategy objects, in v1.1. |
| **R-04** | `print()` everywhere; verbosity is unconfigurable. | Migrate to `logging` in v1.1. |
| **R-05** | `TEAM_COLORS` stores hex strings while OpenCV needs BGR tuples; conversions are scattered. | Centralise a single `bgr_for(team_name)` helper in v1.1. |
| **R-06** | SAM2 batch mode initialises from `frames_dir` and reads any stale JPGs left there from earlier runs. | `run_pipeline_5frames.py` clears `.frames_cache/` at startup; do the same in `run_pipeline_streaming_all.py` in v1.1. |
| **R-07** | No CI. Tests do not run automatically. | Add GitHub Actions running `pytest backend/tests` on push, in v1.1. |
| **R-08** | No automated AC-01..AC-10 evaluation. | Add `test_acceptance.py` in v1.1. |

---

## 13. Out of scope (v1.0)

- Possession tracking, shot detection, foul detection, shot-quality metrics.
- Multi-camera or multi-angle fusion.
- Real-time streaming or low-latency inference.
- Web UI / dashboard / ingestion API. The `frontend/` and `backend/app/api/` skeletons exist but are not part of v1.0.
- Cloud-storage-backed runs (S3 in/out). Local filesystem only.
- Model training inside the pipeline.

---

## 14. Open questions for the PI

These are decisions the spec is currently making by default; flagging for explicit sign-off.

- **Q-01.** Are the AC-03/AC-04 thresholds (≥ 10 player IDs, ≥ 80 % team-named) appropriate for the target data, or should they be tighter?
- **Q-02.** Is "lighter cluster ↔ team A" the right convention, or should we ask the operator to point at one player on frame 0 to seed the assignment?
- **Q-03.** Should v1.1 include possession-segmenting (FR-70 candidate), or stays as an analysis layer above the spec?
- **Q-04.** What's the target environment for stakeholder review — JPG portfolio + MP4 stack as today, or a static HTML report generated from the same data?
- **Q-05.** Is the 25.4 s clip representative? Should we add a 5-minute clip and a full-quarter clip to the regression set?

---

## 15. Roadmap (v1.1 and beyond)

This section is non-binding; it sets direction.

- **v1.1 (next 2 weeks).** Address all R-01..R-08. Single CLI, structured logging, refactor `process_video_with_tracking`, automated acceptance tests, CI on GitHub Actions, fix `get_lighter_cluster` in the wrapper.
- **v1.2.** Possession state machine (FR-70+): who has the ball, transitions on dribble→pass→shot→rebound. Probably needs a ball-tracking module.
- **v1.3.** Web UI minimum: upload an MP4, see the six-stage MP4s and the portfolio. Built on the existing FastAPI scaffold.
- **v2.0.** Multi-game aggregation + per-player metrics dashboard (efficient-shooting %, defensive matchup minutes, transition rate).

---

## Appendix A — Pipeline parameter defaults

These mirror the constants checked into `main` as of 2026-05-02. A change to any of them requires a spec bump to v1.0.x.

All values verified against `main` on 2026-05-02 by direct grep.

| Parameter | Value | Module | Notes |
| --- | --- | --- | --- |
| `DEFAULT_MAX_TOTAL_OBJECTS` | 20 | `player_tracker.py` | |
| `DEFAULT_KEYFRAME_INTERVAL` | 30 | `player_tracker.py` | Module constant. **But function-signature default for `keyframe_interval` is 15** — known v1 inconsistency, see R-04 / NFR-30. |
| `DEFAULT_IOU_THRESHOLD` | 0.3 | `player_tracker.py` | Box-matching IoU when adding new SAM2 prompts mid-stream. |
| `DETECTION_IOU_THRESHOLD` | 0.9 | `player_tracker.py`, `jersey_detector.py` | RF-DETR NMS / jersey-vs-player IoS-style match. Defined in two files — collapse in v1.1. |
| `BYTETRACK_MINIMUM_CONSECUTIVE_FRAMES` | 3 | `player_tracker.py` | |
| `NUMBER_CONSECUTIVE_VALIDATION` | 3 | `player_tracker.py` | |
| `CONSECUTIVE_VALIDATION_FRAMES` | 3 | `jersey_detector.py` | Same intent as above; collapse to one constant in v1.1. |
| `SAM2_MASK_FILTER_RELATIVE_DISTANCE` | 0.03 | `player_tracker.py` | |
| `KMEANS_RANDOM_STATE` | 42 | `team_classifier.py` | |
| `KMEANS_N_INIT` | 10 | `team_classifier.py` | |
| `DEFAULT_JUMP_SIGMA` | 3.5 | `path_smoothing.py` | |
| `DEFAULT_MIN_JUMP_DIST` | 0.6 | `path_smoothing.py` | |
| `DEFAULT_MAX_JUMP_RUN` | 18 | `path_smoothing.py` | |
| `DEFAULT_PAD_AROUND_RUNS` | 2 | `path_smoothing.py` | |
| `DEFAULT_SMOOTH_WINDOW` | 9 | `path_smoothing.py` | |
| `DEFAULT_SMOOTH_POLY` | 2 | `path_smoothing.py` | |
| `DEFAULT_TACTICAL_WINDOW` | 5 | `path_smoothing.py` | Moving-average window for tactical positions. |
| `track_activation_threshold` | 0.15 | `player_referee_detector.py` | ByteTrack arg, not a hoisted constant. v1.1: hoist. |
| `lost_track_buffer` | 600 | `player_referee_detector.py` | Same — not yet hoisted. |
| `minimum_matching_threshold` | 0.5 | `player_referee_detector.py` | Same. |
| `JERSEY_RECOGNITION_INTERVAL` | 5 | `player_tracker.py` (function default) | Not yet a module constant. Hoist in v1.1. |

---

## Appendix B — Change log

| Version | Date | Change |
| --- | --- | --- |
| 1.0.0 | 2026-05-02 | Initial spec covering the system as it lands on `main` after the BFloat16 fp32 fix. |
