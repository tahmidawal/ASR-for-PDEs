#!/bin/bash
#
# Train all Poisson models (lightweight + advanced)
# Run this script to train all models sequentially
#
# Usage: ./train_all.sh [GPU_ID]
#   Example: ./train_all.sh 0
#

echo "========================================"
echo "Training All 1D Poisson Models"
echo "========================================"
echo ""

# Check if GPU argument provided
GPU=${1:-0}
echo "Using GPU: $GPU"
echo ""

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source /cluster/home/tawal01/ASR/venv/bin/activate
fi

# # Dense Net (MLP) - ~395K params
# echo "========================================"
# echo "1/8: Training Dense Net (MLP)"
# echo "========================================"
# python train_poisson1d.py \
#     --config configs/poisson1d_dense.yaml \
#     --name poisson1d_dense \
#     --gpu $GPU

# echo ""
# echo "Dense Net training complete!"
# echo ""

# # Micro CNN - ~11K params
# echo "========================================"
# echo "2/8: Training Micro CNN"
# echo "========================================"
# python train_poisson1d.py \
#     --config configs/poisson1d_micro_cnn.yaml \
#     --name poisson1d_micro_cnn \
#     --gpu $GPU

# echo ""
# echo "Micro CNN training complete!"
# echo ""

# # Nano U-Net - ~8K params
# echo "========================================"
# echo "3/8: Training Nano U-Net"
# echo "========================================"
# python train_poisson1d.py \
#     --config configs/poisson1d_nano_unet.yaml \
#     --name poisson1d_nano_unet \
#     --gpu $GPU

# echo ""
# echo "Nano U-Net training complete!"
# echo ""

# # Nano U-Net Deep - ~40K params
# echo "========================================"
# echo "4/8: Training Nano U-Net Deep"
# echo "========================================"
# python train_poisson1d.py \
#     --config configs/poisson1d_nano_unet_deep.yaml \
#     --name poisson1d_nano_unet_deep \
#     --gpu $GPU

# echo ""
# echo "Nano U-Net Deep training complete!"
# echo ""

# # Better CNN - ~140K params
# echo "========================================"
# echo "5/8: Training Better CNN"
# echo "Expected: 0.1-0.5% relative L2 error"
# echo "========================================"
# python train_poisson1d.py \
#     --config configs/poisson1d_better_cnn.yaml \
#     --name poisson1d_better_cnn \
#     --gpu $GPU

# echo ""
# echo "Better CNN training complete!"
# echo ""

# # UNet Medium - ~538K params
# echo "========================================"
# echo "6/8: Training UNet Medium"
# echo "Expected: 0.05-0.2% relative L2 error"
# echo "========================================"
# python train_poisson1d.py \
#     --config configs/poisson1d_unet_medium.yaml \
#     --name poisson1d_unet_medium \
#     --gpu $GPU

# echo ""
# echo "UNet Medium training complete!"
# echo ""

# FNO-1D - ~543K params
echo "========================================"
echo "7/8: Training FNO-1D ⭐"
echo "Expected: 0.01-0.1% relative L2 error"
echo "Recommended for spectral problems!"
echo "========================================"
python train_poisson1d.py \
    --config configs/poisson1d_fno.yaml \
    --name poisson1d_fno \
    --gpu $GPU

echo ""
echo "FNO-1D training complete!"
echo ""

# # FNO-1D Advanced - ~3.6M params
# echo "========================================"
# echo "8/8: Training FNO-1D Advanced ⭐⭐"
# echo "Expected: 0.001-0.05% relative L2 error"
# echo "Most powerful - targets 0.01% goal!"
# echo "========================================"
# python train_poisson1d.py \
#     --config configs/poisson1d_fno_advanced.yaml \
#     --name poisson1d_fno_advanced \
#     --gpu $GPU

# echo ""
# echo "FNO-1D Advanced training complete!"
# echo ""

# echo "========================================"
# echo "All models trained successfully!"
# echo "========================================"
# echo ""
# echo "Results saved in save/ directory:"
# echo "  - save/poisson1d_dense/"
# echo "  - save/poisson1d_micro_cnn/"
# echo "  - save/poisson1d_nano_unet/"
# echo "  - save/poisson1d_nano_unet_deep/"
# echo "  - save/poisson1d_better_cnn/"
# echo "  - save/poisson1d_unet_medium/"
# echo "  - save/poisson1d_fno/"
# echo "  - save/poisson1d_fno_advanced/"
# echo ""
# echo "To visualize results:"
# echo "  python inference_all_models.py --output_dir inference_results_all"
# echo ""


