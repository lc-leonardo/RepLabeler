#!/bin/bash
# Launch script for RepLabeler
# This script sets up the environment and launches the RepLabeler application

echo "=== RepLabeler - Exercise Video Annotation Tool ==="
echo "Checking dependencies..."

# Check if we're in the right directory
if [ ! -f "rep_labeler.py" ]; then
    echo "Error: rep_labeler.py not found. Please run this script from the RepLabeler directory."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python version: $python_version"

# Check if MMPose is available
python3 -c "import mmpose; print('MMPose: OK')" 2>/dev/null || {
    echo "Warning: MMPose not found. Please install MMPose first:"
    echo "  pip install mmpose"
    echo ""
}

# Check if basic dependencies are available
python3 -c "import tkinter, cv2, PIL, numpy; print('Basic dependencies: OK')" 2>/dev/null || {
    echo "Warning: Some basic dependencies are missing. Please install:"
    echo "  pip install -r requirements.txt"
    echo ""
}

echo "Launching RepLabeler..."

# Check if conda environment exists
if command -v conda &> /dev/null && conda env list | grep -q "openmmlab"; then
    echo "Using openmmlab conda environment..."
    /home/lucas/.miniconda3/bin/conda run -p /home/lucas/.conda/envs/openmmlab --no-capture-output python rep_labeler.py
else
    echo "Using system Python..."
    python3 rep_labeler.py
fi