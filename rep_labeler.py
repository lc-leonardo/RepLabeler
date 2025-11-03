#!/usr/bin/env python3
"""
RepLabeler - A video annotation tool for exercise repetition labeling
Uses MMPose for pose detection and provides an interactive GUI for labeling exercise states
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Add parent directory to path to import MMPose modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from mmpose.apis import inference_topdown, init_model as init_pose_estimator
    from mmdet.apis import inference_detector, init_detector
    from mmpose.evaluation.functional import nms
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import merge_data_samples
    from mmpose.utils import adapt_mmdet_pipeline
    import mmcv
    from mmengine.config import Config
    MMPose_Available = True
except ImportError as e:
    print(f"Warning: MMPose import failed: {e}")
    print("Please ensure MMPose is properly installed")
    MMPose_Available = False
    inference_topdown = None
    init_pose_estimator = None
    init_detector = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RepLabeler:
    """Main application class for exercise repetition labeling."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RepLabeler - Exercise Video Annotation Tool")
        self.root.geometry("1200x800")
        
        # Load configuration
        self.load_config()
        
        # Application state
        self.video_path = None
        self.video_frames = []
        self.pose_results = []
        self.current_frame_idx = 0
        self.frame_labels = {}  # frame_idx -> state_label
        self.total_frames = 0
        
        # MMPose models (following main_metrics.py approach)
        self.detector = None
        self.pose_estimator = None
        self.visualizer = None
        
        # GUI components
        self.video_frame = None
        self.frame_slider = None
        self.state_dropdown = None
        self.frame_counter_label = None
        self.status_label = None
        
        self.setup_gui()
        
    def load_config(self):
        """Load configuration from config.json file."""
        config_path = Path(__file__).parent / "config.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load MMPose settings
            mmpose_config = config.get('mmpose', {})
            self.det_config = mmpose_config.get('det_config', 'MMPose_Models/rtmdet_m_640-8xb32_coco-person.py')
            self.det_checkpoint = mmpose_config.get('det_checkpoint', 'MMPose_Models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth')
            self.pose_config = mmpose_config.get('pose_config', 'MMPose_Models/cspnext-m_udp_8xb64-210e_crowdpose-256x192.py')
            self.pose_checkpoint = mmpose_config.get('pose_checkpoint', 'MMPose_Models/cspnext-m_udp-crowdpose_pt-in1k_210e-256x192-f591079f_20230123.pth')
            self.bbox_thr = mmpose_config.get('bbox_threshold', 0.3)
            self.nms_thr = mmpose_config.get('nms_threshold', 0.3)
            self.kpt_thr = mmpose_config.get('keypoint_threshold', 0.3)
            self.device = mmpose_config.get('device', 'cpu')
            
            # Load exercise states
            self.exercise_states = config.get('exercise_states', [
                "Start Position", "Concentric Phase", "Top Position",
                "Eccentric Phase", "Bottom Position", "Transition", "Rest"
            ])
            
            # Load window size
            app_config = config.get('application', {})
            window_size = app_config.get('window_size', '1200x800')
            self.root.geometry(window_size)
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            # Set default values
            self.det_config = "MMPose_Models/rtmdet_m_640-8xb32_coco-person.py"
            self.det_checkpoint = "MMPose_Models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
            self.pose_config = "MMPose_Models/cspnext-m_udp_8xb64-210e_crowdpose-256x192.py"
            self.pose_checkpoint = "MMPose_Models/cspnext-m_udp-crowdpose_pt-in1k_210e-256x192-f591079f_20230123.pth"
            self.bbox_thr = 0.3
            self.nms_thr = 0.3
            self.kpt_thr = 0.3
            self.device = 'cpu'
            self.exercise_states = [
                "Start Position", "Concentric Phase", "Top Position",
                "Eccentric Phase", "Bottom Position", "Transition", "Rest"
            ]
        
    def setup_gui(self):
        """Initialize the GUI components."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top menu
        self.create_menu()
        
        # Video display area
        self.create_video_display(main_frame)
        
        # Controls area
        self.create_controls(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
        
    def create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Video", command=self.load_video)
        file_menu.add_separator()
        file_menu.add_command(label="Save Labels", command=self.save_labels)
        file_menu.add_command(label="Load Labels", command=self.load_labels)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure States", command=self.configure_states)
        
    def create_video_display(self, parent):
        """Create the video display area."""
        video_frame = ttk.LabelFrame(parent, text="Video Display", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=640, height=480)
        self.video_canvas.pack(expand=True)
        
    def create_controls(self, parent):
        """Create the control panel."""
        controls_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Frame navigation
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frame slider
        ttk.Label(nav_frame, text="Frame:").pack(side=tk.LEFT, padx=(0, 5))
        self.frame_slider = ttk.Scale(nav_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                     command=self.on_frame_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Navigation buttons
        ttk.Button(nav_frame, text="◀◀", command=self.first_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="◀", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="▶", command=self.next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="▶▶", command=self.last_frame).pack(side=tk.LEFT, padx=2)
        
        # Frame counter
        self.frame_counter_label = ttk.Label(nav_frame, text="Frame 0 / 0")
        self.frame_counter_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # State labeling
        label_frame = ttk.Frame(controls_frame)
        label_frame.pack(fill=tk.X)
        
        ttk.Label(label_frame, text="Exercise State:").pack(side=tk.LEFT, padx=(0, 5))
        self.state_var = tk.StringVar(value=self.exercise_states[0])
        self.state_dropdown = ttk.Combobox(label_frame, textvariable=self.state_var, 
                                          values=self.exercise_states, state="readonly")
        self.state_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.state_dropdown.bind('<<ComboboxSelected>>', self.on_state_change)
        
        # Label actions
        ttk.Button(label_frame, text="Apply to Current Frame", 
                  command=self.apply_label).pack(side=tk.LEFT, padx=5)
        ttk.Button(label_frame, text="Apply to Range", 
                  command=self.apply_label_range).pack(side=tk.LEFT, padx=5)
        
    def create_status_bar(self, parent):
        """Create the status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready to load video...")
        self.status_label.pack(side=tk.LEFT)
        
        self.progress_var = tk.StringVar(value="")
        self.progress_label = ttk.Label(status_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.RIGHT)
        
    def load_video(self):
        """Load a video file and process it with MMPose."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        self.video_path = file_path
        self.status_label.config(text="Loading video...")
        self.root.update()
        
        try:
            # Load video frames
            self.load_video_frames()
            
            # Initialize MMPose if not already done
            if self.inferencer is None:
                self.init_mmpose()
            
            # Process video with MMPose
            self.process_video_with_mmpose()
            
            # Update UI
            self.update_frame_slider()
            self.display_current_frame()
            self.status_label.config(text=f"Loaded video: {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            self.status_label.config(text="Error loading video")
            
    def load_video_frames(self):
        """Load all frames from the video."""
        cap = cv2.VideoCapture(self.video_path)
        self.video_frames = []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_frames.append(frame_rgb)
            
            frame_idx += 1
            if frame_idx % 30 == 0:  # Update progress every 30 frames
                self.progress_var.set(f"Loading frames: {frame_idx}/{frame_count}")
                self.root.update()
        
        cap.release()
        self.total_frames = len(self.video_frames)
        self.frame_labels = {}  # Reset frame labels
        self.current_frame_idx = 0
        
        logger.info(f"Loaded {self.total_frames} frames from video")
        
    def init_mmpose(self):
        """Initialize the MMPose inferencer."""
        if not MMPose_Available:
            raise ImportError("MMPose not available")
            
        self.status_label.config(text="Initializing MMPose model...")
        self.root.update()
        
        try:
            # Suppress MMPose logging to prevent terminal flooding
            logging.getLogger('mmengine').setLevel(logging.ERROR)
            logging.getLogger('mmpose').setLevel(logging.ERROR)
            logging.getLogger('mmcv').setLevel(logging.ERROR)
            
            # Get absolute paths for model files
            base_path = Path(__file__).parent.parent
            det_config_path = base_path / self.det_config
            det_checkpoint_path = base_path / self.det_checkpoint
            pose_config_path = base_path / self.pose_config
            pose_checkpoint_path = base_path / self.pose_checkpoint
            
            # Initialize detector
            self.detector = init_detector(
                str(det_config_path), 
                str(det_checkpoint_path), 
                device=self.device
            )
            self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
            
            # Initialize pose estimator
            self.pose_estimator = init_pose_estimator(
                str(pose_config_path),
                str(pose_checkpoint_path),
                device=self.device,
                cfg_options=dict(
                    model=dict(test_cfg=dict(output_heatmaps=False))
                )
            )
            
            # Initialize visualizer
            self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
            self.visualizer.set_dataset_meta(
                self.pose_estimator.dataset_meta, skeleton_style='mmpose'
            )
            
            logger.info("MMPose detector and pose estimator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MMPose: {e}")
            raise
            
    def process_video_with_mmpose(self):
        """Process all video frames with MMPose to extract pose keypoints."""
        if not self.video_frames:
            return
            
        self.status_label.config(text="Processing frames with MMPose...")
        self.root.update()
        
        self.pose_results = []
        
        for i, frame in enumerate(self.video_frames):
            try:
                # Convert RGB to BGR for MMPose
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Run detection to get bboxes
                det_result = inference_detector(self.detector, frame_bgr)
                pred_instance = det_result.pred_instances.cpu().numpy()
                
                # Filter detections by confidence and NMS
                bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                bboxes = bboxes[np.logical_and(pred_instance.labels == 0,  # person class
                                               pred_instance.scores > self.bbox_thr)]
                bboxes = bboxes[nms(bboxes, self.nms_thr), :4]
                
                # Keep the largest bbox (main person in frame)
                if len(bboxes) > 0:
                    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                    max_idx = np.argmax(areas)
                    largest_bbox = bboxes[max_idx:max_idx+1]
                else:
                    largest_bbox = np.empty((0, 4))
                
                # Run pose estimation
                pose_results = inference_topdown(self.pose_estimator, frame_bgr, largest_bbox)
                data_samples = merge_data_samples(pose_results)
                
                # Extract keypoints data
                pred_instances = data_samples.get('pred_instances', None)
                if pred_instances is not None and len(pred_instances) > 0:
                    # Get the first (and typically only) person's keypoints
                    keypoints = pred_instances.keypoints[0].cpu().numpy()  # Shape: (num_keypoints, 3)
                    keypoint_scores = pred_instances.keypoint_scores[0].cpu().numpy()  # Shape: (num_keypoints,)
                    
                    # Combine coordinates and scores
                    result_keypoints = []
                    for j in range(len(keypoints)):
                        x, y = keypoints[j]
                        score = keypoint_scores[j]
                        result_keypoints.append([float(x), float(y), float(score)])
                    
                    result = {
                        'predictions': [{
                            'keypoints': result_keypoints,
                            'bbox': largest_bbox[0].tolist() if len(largest_bbox) > 0 else [0, 0, 0, 0]
                        }]
                    }
                else:
                    result = {'predictions': []}
                    
                self.pose_results.append(result)
                
                if (i + 1) % 10 == 0:  # Update progress every 10 frames
                    self.progress_var.set(f"Processing: {i + 1}/{len(self.video_frames)}")
                    self.root.update()
                    
            except Exception as e:
                logger.warning(f"Failed to process frame {i}: {e}")
                self.pose_results.append({'predictions': []})
        
        self.progress_var.set("")
        logger.info(f"Processed {len(self.pose_results)} frames with MMPose")
        
    def display_current_frame(self):
        """Display the current frame with pose overlay."""
        if not self.video_frames or self.current_frame_idx >= len(self.video_frames):
            return
            
        frame = self.video_frames[self.current_frame_idx].copy()
        
        # Overlay pose if available
        if (self.current_frame_idx < len(self.pose_results) and 
            self.pose_results[self.current_frame_idx] is not None):
            frame = self.draw_pose_overlay(frame, self.pose_results[self.current_frame_idx])
        
        # Resize frame to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Valid dimensions
            frame_pil = Image.fromarray(frame)
            frame_pil = self.resize_image_to_fit(frame_pil, canvas_width, canvas_height)
            
            self.photo = ImageTk.PhotoImage(frame_pil)
            
            self.video_canvas.delete("all")
            self.video_canvas.create_image(canvas_width//2, canvas_height//2, 
                                         anchor=tk.CENTER, image=self.photo)
        
        # Update frame counter and current state
        self.update_frame_info()
        
    def draw_pose_overlay(self, frame, pose_result):
        """Draw pose keypoints and skeleton on the frame."""
        try:
            predictions = pose_result.get('predictions', [])
            if not predictions:
                return frame
                
            for prediction in predictions:
                keypoints = prediction.get('keypoints', [])
                if len(keypoints) == 0:
                    continue
                
                # CrowdPose keypoint indices and skeleton connections
                # Based on CrowdPose 14-keypoint format
                skeleton = [
                    [0, 1], [0, 2], [1, 3], [2, 4],  # Head connections
                    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
                    [5, 11], [6, 12], [11, 12],  # Torso
                    [11, 13], [12, 13]  # Legs (if available)
                ]
                
                # Draw skeleton lines first
                for connection in skeleton:
                    if (connection[0] < len(keypoints) and connection[1] < len(keypoints)):
                        kpt1 = keypoints[connection[0]]
                        kpt2 = keypoints[connection[1]]
                        
                        if (len(kpt1) >= 3 and len(kpt2) >= 3 and 
                            kpt1[2] > self.kpt_thr and kpt2[2] > self.kpt_thr):
                            
                            pt1 = (int(kpt1[0]), int(kpt1[1]))
                            pt2 = (int(kpt2[0]), int(kpt2[1]))
                            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                    
                # Draw keypoints on top
                for i, kpt in enumerate(keypoints):
                    if len(kpt) >= 3:
                        x, y, confidence = int(kpt[0]), int(kpt[1]), kpt[2]
                        
                        if confidence > self.kpt_thr:
                            # Color-code keypoints by body part
                            if i <= 4:  # Head and neck
                                color = (0, 255, 255)  # Yellow
                            elif i <= 10:  # Arms
                                color = (255, 0, 255)  # Magenta  
                            else:  # Torso and legs
                                color = (0, 255, 0)    # Green
                                
                            cv2.circle(frame, (x, y), 4, color, -1)
                            cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)  # Black border
                
        except Exception as e:
            logger.warning(f"Error drawing pose overlay: {e}")
            
        return frame
        
    def resize_image_to_fit(self, image, max_width, max_height):
        """Resize image to fit within the given dimensions while maintaining aspect ratio."""
        img_width, img_height = image.size
        
        # Calculate scaling factor
        scale = min(max_width / img_width, max_height / img_height)
        
        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
        
    def update_frame_slider(self):
        """Update the frame slider configuration."""
        if self.total_frames > 0:
            self.frame_slider.config(from_=0, to=self.total_frames - 1)
            self.frame_slider.set(self.current_frame_idx)
        
    def update_frame_info(self):
        """Update frame counter and state information."""
        self.frame_counter_label.config(
            text=f"Frame {self.current_frame_idx + 1} / {self.total_frames}"
        )
        
        # Update state dropdown to show current frame's label
        current_label = self.frame_labels.get(self.current_frame_idx, self.exercise_states[0])
        self.state_var.set(current_label)
        
    def on_frame_change(self, value):
        """Handle frame slider change."""
        try:
            new_frame_idx = int(float(value))
            if 0 <= new_frame_idx < self.total_frames:
                self.current_frame_idx = new_frame_idx
                self.display_current_frame()
        except ValueError:
            pass
            
    def on_state_change(self, event=None):
        """Handle state dropdown change."""
        # Automatically apply the selected state to current frame
        self.apply_label()
        
    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_slider.set(self.current_frame_idx)
            self.display_current_frame()
            
    def next_frame(self):
        """Go to next frame."""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.frame_slider.set(self.current_frame_idx)
            self.display_current_frame()
            
    def first_frame(self):
        """Go to first frame."""
        if self.total_frames > 0:
            self.current_frame_idx = 0
            self.frame_slider.set(self.current_frame_idx)
            self.display_current_frame()
            
    def last_frame(self):
        """Go to last frame."""
        if self.total_frames > 0:
            self.current_frame_idx = self.total_frames - 1
            self.frame_slider.set(self.current_frame_idx)
            self.display_current_frame()
            
    def apply_label(self):
        """Apply the selected state label to the current frame."""
        if self.total_frames == 0:
            return
            
        state = self.state_var.get()
        self.frame_labels[self.current_frame_idx] = state
        
        labeled_count = len(self.frame_labels)
        self.status_label.config(
            text=f"Labeled {labeled_count}/{self.total_frames} frames"
        )
        
    def apply_label_range(self):
        """Apply the selected state label to a range of frames."""
        if self.total_frames == 0:
            return
            
        # Simple dialog to get range
        dialog = RangeDialog(self.root, self.total_frames, self.current_frame_idx)
        if dialog.result:
            start_frame, end_frame = dialog.result
            state = self.state_var.get()
            
            for i in range(start_frame, end_frame + 1):
                self.frame_labels[i] = state
                
            labeled_count = len(self.frame_labels)
            self.status_label.config(
                text=f"Labeled {labeled_count}/{self.total_frames} frames"
            )
            messagebox.showinfo("Success", 
                              f"Applied '{state}' to frames {start_frame + 1} to {end_frame + 1}")
            
    def configure_states(self):
        """Configure exercise states."""
        dialog = StateConfigDialog(self.root, self.exercise_states)
        if dialog.result:
            self.exercise_states = dialog.result
            self.state_dropdown.config(values=self.exercise_states)
            if self.exercise_states:
                self.state_var.set(self.exercise_states[0])
                
    def save_labels(self):
        """Save frame labels and pose data to JSON file."""
        if not self.video_path:
            messagebox.showwarning("Warning", "No video loaded")
            return
            
        # Check if all frames are labeled
        unlabeled_frames = []
        for i in range(self.total_frames):
            if i not in self.frame_labels:
                unlabeled_frames.append(i + 1)  # 1-based for user display
                
        if unlabeled_frames:
            response = messagebox.askyesno(
                "Unlabeled Frames", 
                f"There are {len(unlabeled_frames)} unlabeled frames. "
                f"Continue saving anyway?\n\nFirst few unlabeled: {unlabeled_frames[:10]}"
            )
            if not response:
                return
        
        # Get save location
        video_name = Path(self.video_path).stem
        default_filename = f"{video_name}_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Labels",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialvalue=default_filename
        )
        
        if not file_path:
            return
            
        try:
            # Prepare data structure
            data = {
                "video_info": {
                    "path": self.video_path,
                    "total_frames": self.total_frames,
                    "det_model": {
                        "config": self.det_config,
                        "checkpoint": self.det_checkpoint
                    },
                    "pose_model": {
                        "config": self.pose_config, 
                        "checkpoint": self.pose_checkpoint
                    },
                    "parameters": {
                        "bbox_threshold": self.bbox_thr,
                        "nms_threshold": self.nms_thr,
                        "keypoint_threshold": self.kpt_thr,
                        "device": self.device
                    },
                    "created_at": datetime.now().isoformat()
                },
                "exercise_states": self.exercise_states,
                "frame_data": []
            }
            
            # Add frame-wise data
            for i in range(self.total_frames):
                frame_data = {
                    "frame_index": i,
                    "state_label": self.frame_labels.get(i, None),
                    "keypoints": None
                }
                
                # Add pose keypoints if available
                if (i < len(self.pose_results) and 
                    self.pose_results[i] is not None):
                    try:
                        predictions = self.pose_results[i].get('predictions', [])
                        if predictions:
                            frame_data["keypoints"] = predictions[0].get('keypoints', [])
                    except:
                        pass
                        
                data["frame_data"].append(frame_data)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            messagebox.showinfo("Success", f"Labels saved to:\n{file_path}")
            self.status_label.config(text=f"Saved labels to {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error saving labels: {e}")
            messagebox.showerror("Error", f"Failed to save labels: {str(e)}")
            
    def load_labels(self):
        """Load previously saved labels."""
        file_path = filedialog.askopenfilename(
            title="Load Labels",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Load exercise states
            if "exercise_states" in data:
                self.exercise_states = data["exercise_states"]
                self.state_dropdown.config(values=self.exercise_states)
                
            # Load frame labels
            self.frame_labels = {}
            frame_data = data.get("frame_data", [])
            
            for frame_info in frame_data:
                frame_idx = frame_info.get("frame_index")
                state_label = frame_info.get("state_label")
                
                if frame_idx is not None and state_label is not None:
                    self.frame_labels[frame_idx] = state_label
                    
            labeled_count = len(self.frame_labels)
            total_frames = data.get("video_info", {}).get("total_frames", "unknown")
            
            self.status_label.config(
                text=f"Loaded {labeled_count}/{total_frames} frame labels"
            )
            self.update_frame_info()
            
            messagebox.showinfo("Success", 
                              f"Loaded labels from:\n{os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            messagebox.showerror("Error", f"Failed to load labels: {str(e)}")
            
    def run(self):
        """Start the application."""
        self.root.mainloop()


class RangeDialog:
    """Dialog for selecting a range of frames."""
    
    def __init__(self, parent, total_frames, current_frame):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Apply Label to Range")
        self.dialog.geometry("400x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        frame = ttk.Frame(self.dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Start frame
        ttk.Label(frame, text="Start Frame:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.start_var = tk.IntVar(value=current_frame + 1)  # 1-based display
        start_spin = ttk.Spinbox(frame, from_=1, to=total_frames, textvariable=self.start_var, width=10)
        start_spin.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # End frame
        ttk.Label(frame, text="End Frame:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.end_var = tk.IntVar(value=current_frame + 1)  # 1-based display
        end_spin = ttk.Spinbox(frame, from_=1, to=total_frames, textvariable=self.end_var, width=10)
        end_spin.grid(row=1, column=1, padx=(10, 0), pady=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)
        
        # Wait for dialog to close
        self.dialog.wait_window()
        
    def ok_clicked(self):
        start = self.start_var.get() - 1  # Convert to 0-based
        end = self.end_var.get() - 1      # Convert to 0-based
        
        if start <= end:
            self.result = (start, end)
        else:
            messagebox.showerror("Invalid Range", "Start frame must be <= End frame")
            return
            
        self.dialog.destroy()
        
    def cancel_clicked(self):
        self.dialog.destroy()


class StateConfigDialog:
    """Dialog for configuring exercise states."""
    
    def __init__(self, parent, current_states):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configure Exercise States")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        frame = ttk.Frame(self.dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Exercise States (one per line):").pack(anchor=tk.W)
        
        self.text_widget = tk.Text(frame, width=40, height=15)
        self.text_widget.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Load current states
        self.text_widget.insert(tk.END, '\n'.join(current_states))
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.RIGHT)
        
        self.dialog.wait_window()
        
    def ok_clicked(self):
        content = self.text_widget.get(1.0, tk.END).strip()
        states = [line.strip() for line in content.split('\n') if line.strip()]
        
        if not states:
            messagebox.showerror("Invalid Input", "Please enter at least one state")
            return
            
        self.result = states
        self.dialog.destroy()
        
    def cancel_clicked(self):
        self.dialog.destroy()


def main():
    """Main entry point."""
    app = RepLabeler()
    app.run()


if __name__ == "__main__":
    main()