# RepLabeler - Exercise Video Annotation Tool

RepLabeler is an interactive GUI application for annotating exercise videos with repetition states. It uses MMPose for pose detection and provides an intuitive interface for frame-by-frame labeling of exercise phases.

## Features

- **Video Loading**: Support for common video formats (MP4, AVI, MOV, MKV, WMV)
- **Pose Detection**: Automatic pose keypoint detection using MMPose with:
  - Detection: `rtmdet_m_640-8xb32_coco-person` (person detection)
  - Pose: `cspnext-m_udp_8xb64-210e_crowdpose-256x192` (pose estimation)
- **Interactive Interface**: 
  - Video display with pose overlay
  - Frame-by-frame navigation with slider and buttons
  - Dropdown selector for exercise states
  - Real-time labeling feedback
- **Flexible Labeling**:
  - Apply labels to individual frames or ranges
  - Customizable exercise states
  - Validation to ensure all frames are labeled
- **Data Export**: Save annotations as JSON files containing:
  - Frame-wise pose keypoints
  - State labels for each frame
  - Video metadata and processing information

## Default Exercise States

The application comes with predefined exercise states that can be customized:

1. **Start Position** - Initial position before movement
2. **Concentric Phase** - Muscle contraction/lifting phase
3. **Top Position** - Peak of the movement
4. **Eccentric Phase** - Muscle lengthening/lowering phase
5. **Bottom Position** - Lowest point of the movement
6. **Transition** - Between repetitions or changing form
7. **Rest** - Pauses or inactive periods

## Installation

### Prerequisites

1. **Python 3.8+**
2. **MMPose** - Follow the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html)
3. **Required Python packages** (see requirements.txt)

### Setup

1. Clone or navigate to the RepJudge repository:
   ```bash
   cd /path/to/RepJudge
   ```

2. Install dependencies:
   ```bash
   pip install -r RepLabeler/requirements.txt
   ```

3. Ensure MMPose is properly installed and the required model is available

## Usage

### Starting the Application

```bash
cd RepLabeler
python rep_labeler.py
```

### Workflow

1. **Load Video**:
   - File → Load Video
   - Select your exercise video file
   - Wait for MMPose processing to complete

2. **Navigate Frames**:
   - Use the slider to jump to specific frames
   - Use navigation buttons: ◀◀ (first), ◀ (previous), ▶ (next), ▶▶ (last)

3. **Label Frames**:
   - Select the appropriate exercise state from the dropdown
   - Click "Apply to Current Frame" or use "Apply to Range" for multiple frames
   - The status bar shows labeling progress

4. **Save Results**:
   - File → Save Labels
   - Choose location and filename for the JSON output

### Tips for Efficient Labeling

- **Use Range Labeling**: For continuous phases (e.g., multiple frames of "Concentric Phase"), use "Apply to Range"
- **Keyboard Navigation**: The interface responds to keyboard focus for quick navigation
- **Progress Tracking**: The status bar always shows how many frames are labeled vs. total
- **Validation**: The app will warn you about unlabeled frames before saving

## Output Format

The saved JSON file contains:

```json
{
  "video_info": {
    "path": "/path/to/video.mp4",
    "total_frames": 150,
    "model_used": "cspnext-m_udp_8xb64-210e_crowdpose-256x192",
    "created_at": "2025-09-23T10:30:00"
  },
  "exercise_states": ["Start Position", "Concentric Phase", ...],
  "frame_data": [
    {
      "frame_index": 0,
      "state_label": "Start Position",
      "keypoints": [[x1, y1, confidence1], [x2, y2, confidence2], ...]
    },
    ...
  ]
}
```

### Keypoint Format

- Each keypoint is [x, y, confidence]
- Coordinates are in pixels relative to the original video frame
- Confidence values range from 0.0 to 1.0
- Keypoint order follows the CrowdPose dataset format

## Customization

### Exercise States

1. Settings → Configure States
2. Edit the list (one state per line)
3. Save changes

### Model Configuration

To use different MMPose models, modify the model paths in `rep_labeler.py`:

```python
# Detection model
self.det_config = "MMPose_Models/your-detection-config.py"
self.det_checkpoint = "MMPose_Models/your-detection-checkpoint.pth"

# Pose estimation model  
self.pose_config = "MMPose_Models/your-pose-config.py"
self.pose_checkpoint = "MMPose_Models/your-pose-checkpoint.pth"
```

Or edit `config.json` to change the default models.

## Troubleshooting

### Common Issues

1. **MMPose Import Error**:
   - Ensure MMPose is installed: `pip install mmpose`
   - Check that MMEngine and MMCV are compatible versions

2. **Model Loading Fails**:
   - The app will automatically download the required model on first use
   - Ensure internet connection for model download

3. **Video Loading Issues**:
   - Check that OpenCV is installed: `pip install opencv-python`
   - Try converting video to MP4 format if other formats fail

4. **Performance Issues**:
   - Large videos may take time to process
   - Consider resizing videos if memory issues occur

### Logs

The application logs important events to the console. For debugging, check the terminal output for error messages.

## Development

### Project Structure

```
RepLabeler/
├── rep_labeler.py      # Main application
├── requirements.txt    # Python dependencies  
└── README.md          # This documentation
```

### Key Classes

- **RepLabeler**: Main application class
- **RangeDialog**: Dialog for selecting frame ranges
- **StateConfigDialog**: Dialog for customizing exercise states

### Extension Points

- Add new exercise types by modifying `exercise_states`
- Enhance pose visualization in `draw_pose_overlay()`
- Add export formats by extending `save_labels()`

## License

This project is part of the RepJudge repository. Please refer to the main repository license.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check this documentation
2. Review the console output for error messages
3. Open an issue in the RepJudge repository