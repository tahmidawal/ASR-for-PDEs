#!/bin/bash
#
# Run inference on all trained models
#
# Usage:
#   bash run_inference.sh                    # Use defaults
#   bash run_inference.sh --num_samples 10   # Generate 10 test samples

echo "========================================"
echo "Running Inference on All Models"
echo "========================================"
echo ""

# Activate virtual environment if needed
if [ -d "/cluster/home/tawal01/ASR/venv" ]; then
    echo "Activating virtual environment..."
    source /cluster/home/tawal01/ASR/venv/bin/activate
fi

cd /cluster/home/tawal01/ASR/2025-10-18

# Run inference
python inference_all_models.py \
    --save_dir save \
    --output_dir inference_results \
    --num_samples 5 \
    --seed 12345 \
    --device cuda \
    "$@"

echo ""
echo "========================================"
echo "Inference Complete!"
echo "========================================"
echo ""
echo "Results saved in: inference_results/"
echo ""
echo "Generated plots:"
echo "  - inference_results/poisson1d_dense_inference.png"
echo "  - inference_results/poisson1d_micro_cnn_inference.png"
echo "  - inference_results/poisson1d_nano_unet_inference.png"
echo "  - inference_results/poisson1d_nano_unet_deep_inference.png"
echo "  - inference_results/all_models_comparison.png"
echo ""

