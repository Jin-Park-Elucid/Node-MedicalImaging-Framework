#!/bin/bash
# Alternative activation script for manual use
# Usage: source activate.sh

# Activate virtual environment
source venv/bin/activate

# Set project root
export PROJECT_ROOT="$PWD"

# Add to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Optional environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONUNBUFFERED=1

echo "✅ Medical Imaging Framework environment activated"
echo "📁 Project root: $PROJECT_ROOT"
echo "🐍 Python: $(python --version)"
echo "📦 Virtual env: venv/"
echo ""
echo "Available commands:"
echo "  python examples/simple_test.py          - Run quick test"
echo "  python examples/segmentation_workflow.py - Run example workflow"
echo "  python -m medical_imaging_framework.gui.editor - Launch GUI"
echo ""
echo "Documentation:"
echo "  cat docs/INDEX.md                       - Documentation index"
echo "  cat docs/GETTING_STARTED.md            - Quick start guide"
echo ""
