#!/bin/bash
#SBATCH --job-name=lightweight_poisson
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Ultra-Lightweight 1D Poisson Training
# Trains all 4 models: Dense Net, Micro CNN, Nano U-Net, Nano U-Net Deep
#
# Submit with: sbatch submit_training.sh

echo "Starting ultra-lightweight Poisson training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

# Load necessary modules (adjust for your cluster)
# Uncomment and modify as needed:
# module load python/3.9
# module load cuda/11.8
# module load pytorch/2.0

# Activate virtual environment if available
if [ -d "/cluster/home/tawal01/ASR/venv" ]; then
    echo "Activating virtual environment..."
    source /cluster/home/tawal01/ASR/venv/bin/activate
fi

# Change to working directory
cd /cluster/home/tawal01/ASR/2025-10-18

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================"
echo "Environment Info"
echo "========================================"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Train all models
bash train_all.sh 0

echo ""
echo "Training completed at: $(date)"
echo "Job ID: $SLURM_JOB_ID"


