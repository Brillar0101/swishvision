# SwishVision Tests

Isolated tests for specific pipeline components. Each test can be run independently to validate individual features.

## Test Files

### test_player_tracking.py
**What it tests:** Player detection and tracking consistency
**Components:** RF-DETR detection + ByteTrack tracking only
**Output:**
- Visualization video with tracker IDs
- Tracking quality metrics report
- Analysis of ID persistence and switching

**Use case:** Verify that players are detected consistently throughout the video and that ByteTrack maintains stable IDs.

**Run:**
```bash
cd backend/tests
python3 test_player_tracking.py
```

**Metrics analyzed:**
- Total unique tracker IDs
- Persistent trackers (appear in >80% of lifetime, >30 frames)
- Short-lived trackers (ID switches, noise)
- Detection consistency (players per frame)
- Frame coverage (frames with no detections)

---

### test_jersey_detection.py
**What it tests:** Full pipeline with jersey number recognition
**Components:** RF-DETR + ByteTrack + SAM2 + Team Classification + Jersey OCR
**Output:** Full annotated video with jersey numbers and team colors

**Use case:** Test end-to-end pipeline including jersey number recognition.

**Run:**
```bash
cd backend/tests
python3 test_jersey_detection.py
```

---

### test_tactical_pipeline.py
**What it tests:** Tactical view generation
**Components:** Court detection + homography + 2D court visualization
**Output:** Video with tactical overlay

**Use case:** Verify court keypoint detection and tactical view transformation.

**Run:**
```bash
cd backend/tests
python3 test_tactical_pipeline.py
```

---

### test_homography.py
**What it tests:** Homography transformation accuracy
**Components:** Court keypoint detection + perspective transformation
**Output:** Visualization of keypoint matching

**Use case:** Debug court detection and transformation issues.

**Run:**
```bash
cd backend/tests
python3 test_homography.py
```

---

## Test Output Structure

```
backend/tests/outputs/
├── tracking_test/
│   ├── tracking_test.mp4           # Visualization with tracker IDs
│   └── tracking_metrics.txt        # Detailed tracking report
├── jersey_test/
│   └── ...
└── tactical_test/
    └── ...
```

## Troubleshooting

### Issue: Players losing tracking when they re-enter frame
**Test to run:** `test_player_tracking.py`
**Look for:** High number of short-lived trackers, low persistent tracker count
**Expected:** 8-12 persistent trackers for basketball game

### Issue: Jersey numbers not detected
**Test to run:** `test_jersey_detection.py`
**Check:** Roboflow API key, OCR model loading
**Expected:** Jersey numbers appear after 3 consecutive frame validations

### Issue: Tactical view empty or incorrect
**Test to run:** `test_tactical_pipeline.py` or `test_homography.py`
**Look for:** Court keypoint detection failures, homography errors
**Expected:** Court transformer builds successfully on first frame
