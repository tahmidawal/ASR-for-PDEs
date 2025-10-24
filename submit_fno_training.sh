#!/bin/bash
#SBATCH --job-name=fno_poisson_official
#SBATCH --output=logs/fno_train_%j.out
#SBATCH --error=logs/fno_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Official FNO Implementation for 1D Poisson Equation
# This uses our custom FNO implementation following the original paper
# Expected to achieve much better accuracy than existing models
#
# Submit with: sbatch submit_fno_training.sh

echo "Starting Official FNO Training for 1D Poisson Equation"
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

echo "========================================"
echo "Testing FNO Model Integration"
echo "========================================"
python -c "
import models
import torch
print('Testing FNO model creation...')
model = models.make({'name': 'fno-1d-official', 'args': {'modes': 24, 'width': 96, 'depth': 6}})
print(f'✓ FNO model created successfully!')
print(f'✓ Parameters: {sum(p.numel() for p in model.parameters()):,}')

# Test forward pass
x = torch.randn(2, 1, 256)
y = model(x)
print(f'✓ Input shape: {x.shape}')
print(f'✓ Output shape: {y.shape}')
print('✓ Forward pass successful!')
print('✓ All tests passed - ready for training!')
"

if [ $? -ne 0 ]; then
    echo "❌ Model test failed! Exiting..."
    exit 1
fi

echo ""
echo "========================================"
echo "Starting FNO Training"
echo "========================================"

# Train the official FNO model
python train_poisson1d.py \
    --config configs/poisson1d_fno_official.yaml \
    --name poisson1d_fno_official \
    --gpu 0

echo ""
echo "========================================"
echo "Training Summary"
echo "========================================"
echo "Training completed at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Results saved in: save/poisson1d_fno_official/"
echo ""
echo "Check results with:"
echo "  - Training log: save/poisson1d_fno_official/log.txt"
echo "  - Best model: save/poisson1d_fno_official/epoch-best.pth"
echo "  - Metrics plot: save/poisson1d_fno_official/training_metrics.png"
