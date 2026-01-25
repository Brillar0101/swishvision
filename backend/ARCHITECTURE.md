# SwishVision Architecture

## Component Overview

### PlayerRefereeDetector (player_referee_detector.py)
**Purpose:** Low-level detection and tracking only

**What it does:**
- RF-DETR player/referee detection (or Roboflow API fallback)
- ByteTrack for persistent tracker IDs
- Returns `sv.Detections` with bounding boxes and tracker IDs

**What it does NOT do:**
- No segmentation (SAM2)
- No team classification
- No jersey number recognition
- No video generation
- No tactical view

**Usage:**
```python
detector = PlayerRefereeDetector()
detections = detector.detect_and_track(frame)  # Returns detections with tracker IDs
```

---

### PlayerTracker (player_tracker.py)
**Purpose:** High-level pipeline orchestrator

**What it does:**
- Uses `PlayerRefereeDetector` for detection + tracking
- Adds SAM2 segmentation (pixel-perfect masks)
- Adds team classification (SigLIP + K-means)
- Adds jersey number detection (OCR)
- Adds tactical view (court transformation)
- Generates 6-stage portfolio videos
- Handles checkpointing/resume
- Coordinates entire pipeline

**Pipeline stages:**
1. Detection + Tracking (ByteTrack IDs)
2. SAM2 Segmentation (pixel masks)
3. Team Classification (color-coded)
4. Jersey Numbers (OCR with validation)
5. Court Overlay
6. Tactical View (2D court)

**Usage:**
```python
tracker = PlayerTracker()
result = tracker.process_video_with_tracking(
    video_path='video.mp4',
    use_bytetrack=True,
    use_sam2_segmentation=True,
    # ... many more options
)
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    PlayerTracker                         │
│              (Pipeline Orchestrator)                      │
│                                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 1. Detection & Tracking                            │  │
│  │    Uses: PlayerRefereeDetector                     │  │
│  │    Output: Bounding boxes + tracker IDs            │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 2. SAM2 Segmentation (optional)                    │  │
│  │    Input: Bounding boxes from step 1               │  │
│  │    Output: Pixel-perfect masks                     │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 3. Team Classification                             │  │
│  │    Uses: TeamClassifier (SigLIP + K-means)         │  │
│  │    Output: Team assignments (0 or 1)               │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 4. Jersey Detection (optional)                     │  │
│  │    Uses: JerseyDetector (RF-DETR + SmolVLM2)       │  │
│  │    Output: Jersey numbers per player               │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 5. Court Detection & Tactical View                 │  │
│  │    Uses: CourtDetector + TacticalView              │  │
│  │    Output: 2D court positions                      │  │
│  └───────────────────────────────────────────────────┘  │
│                          ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 6. Video Generation                                │  │
│  │    Output: 6 annotated videos (one per stage)      │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

               ┌──────────────────────────┐
               │ PlayerRefereeDetector    │
               │ (Detection Component)    │
               │                          │
               │ - RF-DETR detection      │
               │ - ByteTrack tracking     │
               │ - Returns sv.Detections  │
               └──────────────────────────┘
```

---

## When to Use Each Component

### Use PlayerRefereeDetector directly when:
- Testing detection quality only
- Testing tracking persistence only
- Building custom pipelines
- Need just bounding boxes + tracker IDs
- Want maximum control and minimal overhead

**Example:** The test file `tests/test_player_tracking.py` uses `PlayerRefereeDetector` directly.

### Use PlayerTracker when:
- Running the full SwishVision pipeline
- Need segmentation, teams, jerseys, tactical view
- Generating portfolio videos
- Want end-to-end processing with checkpointing

**Example:** The pipeline runner `run_pipeline_local.py` uses `PlayerTracker`.

---

## Is PlayerTracker Redundant?

**No!** They serve different purposes:

- **PlayerRefereeDetector** = Core detection/tracking engine (reusable component)
- **PlayerTracker** = Full pipeline orchestrator (uses detector + adds everything else)

**Analogy:**
- `PlayerRefereeDetector` is like a car engine
- `PlayerTracker` is like the complete car (engine + transmission + wheels + electronics + body)

You can test the engine independently, but you need the full car for actual driving.

---

## File Naming

Current naming is appropriate:
- `player_referee_detector.py` - Detects players and referees
- `player_tracker.py` - Tracks players through full pipeline (including detection, segmentation, classification, etc.)

**No rename needed** - the names accurately reflect their purposes.
