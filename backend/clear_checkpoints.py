#!/usr/bin/env python3
"""
Clear corrupted checkpoints from a specific output directory.
"""
import shutil
from pathlib import Path
import sys

def clear_checkpoints(output_dir: str):
    """Delete all checkpoint data for a pipeline run."""
    checkpoint_dir = Path(output_dir) / '.checkpoints'
    frames_dir = Path(output_dir) / '.frames_cache'

    if checkpoint_dir.exists():
        print(f"Deleting: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
        print("✓ Checkpoints cleared")
    else:
        print(f"No checkpoints found at: {checkpoint_dir}")

    if frames_dir.exists():
        print(f"Deleting: {frames_dir}")
        shutil.rmtree(frames_dir)
        print("✓ Frame cache cleared")
    else:
        print(f"No frame cache found at: {frames_dir}")

    print("\nYou can now run the pipeline fresh with resume=False")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "portfolio_outputs_full"

    print(f"Clearing checkpoints for: {output_dir}")
    clear_checkpoints(output_dir)
