#!/bin/bash
# Launch script for Video Pose Repetition Labeller
# Simple launcher for the video annotation tool

echo "=== Video Pose Repetition Labeller ==="
echo "Launching annotation tool..."

# Check if we're in the right directory
if [ ! -f "video_pose_labeler.py" ]; then
    echo "Error: video_pose_labeler.py not found. Please run this script from the RepLabeler directory."
    exit 1
fi

# Check basic dependencies
python3 -c "import tkinter, cv2, PIL; print('Dependencies: OK')" 2>/dev/null || {
    echo "Warning: Dependencies missing. Please install:"
    echo "  pip install -r requirements.txt"
    echo ""
}

# Launch the application
python3 video_pose_labeler.py