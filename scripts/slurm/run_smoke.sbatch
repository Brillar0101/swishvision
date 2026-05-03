#!/bin/bash
# Smoke run: 5 random frames × 6 cumulative-stage JPGs (no MP4 output).
# Submit with: sbatch scripts/slurm/run_smoke.sbatch
#SBATCH --account=cjones_swish
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=swish_smoke
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.out

set -euo pipefail

source .venv/bin/activate
cd "$(dirname "$(realpath "$0")")/../.."

python -m swishvision smoke --run "smoke_${SLURM_JOB_ID}" --max-seconds 8
