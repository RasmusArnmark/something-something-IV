#!/bin/bash
# Script to run training with proper environment and logging

cd /Users/rasmusarnmark/Desktop/skku/IV/final_project/something-something-IV

echo "Starting training at $(date)"
echo "====================================="

# Run with conda and capture output
conda run -n sthsth python scripts/train_2d.py \
    --config configs/config_2d_test.yaml \
    --device cpu 2>&1 | tee training.log

echo "====================================="
echo "Training finished at $(date)"
