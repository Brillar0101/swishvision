#!/bin/bash
#SBATCH --job-name=rfdetr-basketball
#SBATCH --partition=h200_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=rfdetr_%j.log
#SBATCH --error=rfdetr_%j.err

# ============================================================
# RF-DETR Training on VT ARC Tinkercliffs
#
# Usage:
#   # First, create .env file with your secrets:
#   # echo 'ROBOFLOW_API_KEY=your_key' >> backend/.env
#   # echo 'SLURM_ACCOUNT=your_allocation' >> backend/.env
#
#   # Then submit with:
#   source backend/.env && sbatch --account=$SLURM_ACCOUNT backend/app/ml/train_rfdetr_cluster.sh
# ============================================================

echo "Starting RF-DETR training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.1.1

# Create and activate conda environment (first time only)
if [ ! -d "$HOME/envs/rfdetr" ]; then
    echo "Creating conda environment..."
    conda create -n rfdetr python=3.11 -y
fi

source activate rfdetr

# Install dependencies (first time only)
pip install --quiet rfdetr roboflow python-dotenv torch torchvision

# Navigate to project directory
cd $HOME/swishvision/backend/app/ml

# Load environment variables from .env file
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
    echo "Create one with:"
    echo "  echo 'ROBOFLOW_API_KEY=your_key' > ../. env"
    echo "  echo 'SLURM_ACCOUNT=your_allocation' >> ../.env"
    exit 1
fi

# Run training (H200 can handle larger batch size)
echo "Starting training..."
python train_rfdetr.py \
    --epochs 50 \
    --batch-size 32 \
    --image-size 560 \
    --output output/rfdetr_basketball

echo "Training complete!"
date
