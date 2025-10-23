#!/bin/bash
#
# Train all ultra-lightweight models
# Run this script to train all 4 models sequentially
#

echo "========================================"
echo "Training Ultra-Lightweight Poisson Models"
echo "========================================"
echo ""

# Check if GPU argument provided
GPU=${1:-0}
echo "Using GPU: $GPU"
echo ""

# Dense Net (MLP) - ~8K params
echo "========================================"
echo "1/4: Training Dense Net (MLP)"
echo "========================================"
python train_poisson1d.py \
    --config configs/poisson1d_dense.yaml \
    --name poisson1d_dense \
    --gpu $GPU

echo ""
echo "Dense Net training complete!"
echo ""

# Micro CNN - ~11K params
echo "========================================"
echo "2/4: Training Micro CNN"
echo "========================================"
python train_poisson1d.py \
    --config configs/poisson1d_micro_cnn.yaml \
    --name poisson1d_micro_cnn \
    --gpu $GPU

echo ""
echo "Micro CNN training complete!"
echo ""

# Nano U-Net - ~25K params
echo "========================================"
echo "3/4: Training Nano U-Net"
echo "========================================"
python train_poisson1d.py \
    --config configs/poisson1d_nano_unet.yaml \
    --name poisson1d_nano_unet \
    --gpu $GPU

echo ""
echo "Nano U-Net training complete!"
echo ""

# Nano U-Net Deep - ~40K params
echo "========================================"
echo "4/4: Training Nano U-Net Deep"
echo "========================================"
python train_poisson1d.py \
    --config configs/poisson1d_nano_unet_deep.yaml \
    --name poisson1d_nano_unet_deep \
    --gpu $GPU

echo ""
echo "Nano U-Net Deep training complete!"
echo ""

echo "========================================"
echo "All models trained successfully!"
echo "========================================"
echo ""
echo "Results saved in save/ directory:"
echo "  - save/poisson1d_dense/"
echo "  - save/poisson1d_micro_cnn/"
echo "  - save/poisson1d_nano_unet/"
echo "  - save/poisson1d_nano_unet_deep/"
echo ""


