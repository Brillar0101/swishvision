#!/bin/bash
# Full SwishVision pipeline against the default test video on a single A100.
# Submit with: sbatch scripts/slurm/run_full.sbatch
#SBATCH --account=cjones_swish
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=swish_full
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.out

set -euo pipefail

# Activate the project venv (created with `python3 -m venv .venv && pip install -e .`)
source .venv/bin/activate

# Anchor working directory at the repo root regardless of where sbatch was invoked
cd "$(dirname "$(realpath "$0")")/../.."

python -m swishvision full --run "run_${SLURM_JOB_ID}"
