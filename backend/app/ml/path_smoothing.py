"""
Path Smoothing Utilities for Tactical View.
Based on Roboflow's clean_paths implementation.

Handles:
- Detecting sudden jumps in player positions
- Removing short abnormal position runs
- Linear interpolation for missing segments
- Savitzky-Golay smoothing for natural movement
"""
import numpy as np
from typing import Tuple, Optional
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


def detect_jumps(
    xy: np.ndarray,
    sigma: float = 3.5,
    min_dist: float = 0.6,
) -> np.ndarray:
    """
    Detect sudden jumps in position using speed analysis.

    Args:
        xy: Position array of shape (n_frames, 2)
        sigma: Number of standard deviations for outlier detection
        min_dist: Minimum distance to consider a jump

    Returns:
        Boolean mask where True indicates a jump
    """
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=bool)

    # Calculate frame-to-frame distances
    diffs = np.diff(xy, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    # Use robust statistics (median absolute deviation)
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    threshold = median_dist + sigma * 1.4826 * mad  # 1.4826 scales MAD to std

    # Also enforce minimum distance
    threshold = max(threshold, min_dist)

    # Mark frames after a jump
    jump_mask = np.zeros(len(xy), dtype=bool)
    jump_mask[1:] = distances > threshold

    return jump_mask


def expand_mask_runs(
    mask: np.ndarray,
    max_run_length: int,
    pad: int,
) -> np.ndarray:
    """
    Remove short runs and pad around them.

    Args:
        mask: Boolean mask of detected issues
        max_run_length: Maximum length of runs to remove entirely
        pad: Number of frames to pad around each removed run

    Returns:
        Expanded boolean mask
    """
    if not np.any(mask):
        return mask

    result = mask.copy()

    # Find runs of True values
    changes = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        run_length = end - start
        if run_length <= max_run_length:
            # Expand the run with padding
            pad_start = max(0, start - pad)
            pad_end = min(len(mask), end + pad)
            result[pad_start:pad_end] = True

    return result


def interpolate_missing(
    xy: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Fill missing segments with linear interpolation.

    Args:
        xy: Position array of shape (n_frames, 2)
        mask: Boolean mask where True indicates missing/bad data

    Returns:
        Interpolated position array
    """
    result = xy.copy()

    for dim in range(2):
        values = result[:, dim]
        valid_idx = np.where(~mask)[0]

        if len(valid_idx) < 2:
            continue

        invalid_idx = np.where(mask)[0]
        if len(invalid_idx) == 0:
            continue

        # Linear interpolation
        result[invalid_idx, dim] = np.interp(
            invalid_idx,
            valid_idx,
            values[valid_idx]
        )

    return result


def smooth_path(
    xy: np.ndarray,
    window: int = 9,
    poly_order: int = 2,
) -> np.ndarray:
    """
    Smooth path using Savitzky-Golay filter.

    Args:
        xy: Position array of shape (n_frames, 2)
        window: Window size for filter (must be odd)
        poly_order: Polynomial order for filter

    Returns:
        Smoothed position array
    """
    if len(xy) < window:
        return xy

    result = xy.copy()
    for dim in range(2):
        result[:, dim] = savgol_filter(
            result[:, dim],
            window_length=window,
            polyorder=poly_order
        )

    return result


def clean_paths(
    video_xy: np.ndarray,
    jump_sigma: float = 3.5,
    min_jump_dist: float = 0.6,
    max_jump_run: int = 18,
    pad_around_runs: int = 2,
    smooth_window: int = 9,
    smooth_poly: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean player movement paths by removing teleportation artifacts.

    This is the main function from Roboflow's notebook that:
    1. Detects sudden position jumps using robust speed analysis
    2. Removes short abnormal runs and nearby frames
    3. Fills missing segments with linear interpolation
    4. Smooths all paths with Savitzky-Golay filter

    Args:
        video_xy: Position array of shape (n_frames, n_players, 2)
        jump_sigma: Std deviations for jump detection
        min_jump_dist: Minimum distance to consider a jump
        max_jump_run: Maximum length of runs to remove
        pad_around_runs: Padding around removed runs
        smooth_window: Window size for smoothing
        smooth_poly: Polynomial order for smoothing

    Returns:
        Tuple of (cleaned_xy, edited_mask)
        - cleaned_xy: Cleaned position array
        - edited_mask: Boolean mask of edited frames
    """
    n_frames, n_players, _ = video_xy.shape
    cleaned_xy = np.zeros_like(video_xy)
    edited_mask = np.zeros((n_frames, n_players), dtype=bool)

    for player_idx in range(n_players):
        xy = video_xy[:, player_idx, :].copy()

        # Detect jumps
        jump_mask = detect_jumps(xy, jump_sigma, min_jump_dist)

        # Expand mask to remove short runs
        expanded_mask = expand_mask_runs(jump_mask, max_jump_run, pad_around_runs)

        # Interpolate missing segments
        xy = interpolate_missing(xy, expanded_mask)

        # Smooth the path
        if smooth_window > 0 and len(xy) >= smooth_window:
            xy = smooth_path(xy, smooth_window, smooth_poly)

        cleaned_xy[:, player_idx, :] = xy
        edited_mask[:, player_idx] = expanded_mask

    return cleaned_xy, edited_mask


def smooth_tactical_positions(
    positions_history: list,
    window_size: int = 5,
) -> list:
    """
    Simple moving average smoothing for tactical view positions.

    Args:
        positions_history: List of dicts mapping obj_id -> (x, y)
        window_size: Size of smoothing window

    Returns:
        Smoothed positions history
    """
    if len(positions_history) < window_size:
        return positions_history

    # Get all object IDs
    all_ids = set()
    for pos_dict in positions_history:
        all_ids.update(pos_dict.keys())

    # Convert to arrays for each object
    smoothed = []
    for frame_idx in range(len(positions_history)):
        smoothed_dict = {}

        for obj_id in all_ids:
            # Collect positions in window
            start_idx = max(0, frame_idx - window_size // 2)
            end_idx = min(len(positions_history), frame_idx + window_size // 2 + 1)

            x_vals, y_vals = [], []
            for i in range(start_idx, end_idx):
                if obj_id in positions_history[i]:
                    x, y = positions_history[i][obj_id]
                    x_vals.append(x)
                    y_vals.append(y)

            if x_vals:
                smoothed_dict[obj_id] = (np.mean(x_vals), np.mean(y_vals))

        smoothed.append(smoothed_dict)

    return smoothed
