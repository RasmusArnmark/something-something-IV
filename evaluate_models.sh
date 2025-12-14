#!/bin/bash

# Comprehensive Model Evaluation Script
# Evaluates all three 3D models and generates beautiful visualizations

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║           🎯 3D Model Evaluation Suite - Something-Something V2           ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
NUM_SAMPLES=${1:-1000}  # Default to 1000 samples, or use first argument
DEVICE=${2:-mps}         # Default to mps, or use second argument
OUTPUT_DIR="evaluation_results"

echo "📋 Configuration:"
echo "   • Number of samples: $NUM_SAMPLES"
echo "   • Device: $DEVICE"
echo "   • Output directory: $OUTPUT_DIR"
echo ""

# Check if checkpoints exist
echo "🔍 Checking for model checkpoints..."
if [ ! -f "ssh_checkpoints/checkpoints/best_3d_model.pth" ]; then
    echo "❌ Error: From-scratch checkpoint not found!"
    exit 1
fi
if [ ! -f "ssh_checkpoints/checkpoints_pretrained/best_3d_model.pth" ]; then
    echo "❌ Error: Kinetics pretrained checkpoint not found!"
    exit 1
fi
if [ ! -f "ssh_checkpoints/checkpoints_2d_inflated/best_3d_model.pth" ]; then
    echo "❌ Error: 2D-inflated checkpoint not found!"
    exit 1
fi
echo "✅ All checkpoints found!"
echo ""

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import matplotlib" 2>/dev/null || {
    echo "Installing matplotlib..."
    pip install matplotlib seaborn pandas
}
echo "✅ All dependencies satisfied!"
echo ""

# Run evaluation
echo "🚀 Starting evaluation..."
echo ""

python scripts/evaluate_all_models.py \
    --num_samples $NUM_SAMPLES \
    --device $DEVICE \
    --output_dir $OUTPUT_DIR

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          ✨ EVALUATION COMPLETE! ✨                         ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📁 Generated files:"
    echo "   📊 ${OUTPUT_DIR}/accuracy_comparison.png"
    echo "   📊 ${OUTPUT_DIR}/success_failure.png"
    echo "   📊 ${OUTPUT_DIR}/From_Scratch_per_class.png"
    echo "   📊 ${OUTPUT_DIR}/Kinetics_Pretrained_per_class.png"
    echo "   📊 ${OUTPUT_DIR}/2D-Inflated_per_class.png"
    echo "   💾 ${OUTPUT_DIR}/results.json"
    echo ""
    echo "🖼️  Opening results..."
    
    # Open the comparison chart (macOS)
    if command -v open &> /dev/null; then
        open ${OUTPUT_DIR}/accuracy_comparison.png 2>/dev/null || true
    fi
    
    echo ""
    echo "💡 Tip: View all charts in ${OUTPUT_DIR}/"
    echo ""
else
    echo ""
    echo "❌ Evaluation failed! Check the error messages above."
    exit 1
fi
