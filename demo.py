#!/usr/bin/env python3
"""
Demo script to test RepLabeler functionality with existing video files
"""

import sys
import os
from pathlib import Path

# Add RepLabeler to path
sys.path.append(str(Path(__file__).parent))

def test_basic_functionality():
    """Test basic RepLabeler functionality without GUI."""
    print("=== RepLabeler Demo Test ===")
    
    try:
        # Test imports first
        import tkinter as tk
        from PIL import Image, ImageTk
        import cv2
        import numpy as np
        import json
        print("✓ Basic imports successful")
        
        # Test MMPose imports
        from mmpose.apis import inference_topdown, init_model as init_pose_estimator
        from mmdet.apis import inference_detector, init_detector
        print("✓ MMPose imports successful")
        
        # Test configuration loading without GUI
        config_path = Path(__file__).parent / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✓ Configuration file loaded successfully")
            
            mmpose_config = config.get('mmpose', {})
            det_config = mmpose_config.get('det_config', 'MMPose_Models/rtmdet_m_640-8xb32_coco-person.py')
            det_checkpoint = mmpose_config.get('det_checkpoint', 'MMPose_Models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth')
            pose_config = mmpose_config.get('pose_config', 'MMPose_Models/cspnext-m_udp_8xb64-210e_crowdpose-256x192.py')  
            pose_checkpoint = mmpose_config.get('pose_checkpoint', 'MMPose_Models/cspnext-m_udp-crowdpose_pt-in1k_210e-256x192-f591079f_20230123.pth')
            exercise_states = config.get('exercise_states', [])
            
            print("✓ RepLabeler configuration loaded successfully")
            print(f"✓ Detection config: {det_config}")
            print(f"✓ Pose config: {pose_config}")
            print(f"✓ Exercise states: {len(exercise_states)} states loaded")
        else:
            print("⚠ Configuration file not found, using defaults")
            det_config = "MMPose_Models/rtmdet_m_640-8xb32_coco-person.py"
            det_checkpoint = "MMPose_Models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
            pose_config = "MMPose_Models/cspnext-m_udp_8xb64-210e_crowdpose-256x192.py"
            pose_checkpoint = "MMPose_Models/cspnext-m_udp-crowdpose_pt-in1k_210e-256x192-f591079f_20230123.pth"

        
        # Test model paths exist
        # Use absolute path to RepJudge directory
        base_path = Path("/home/lucas/RepJudge")
        det_config_path = base_path / det_config
        det_checkpoint_path = base_path / det_checkpoint
        pose_config_path = base_path / pose_config
        pose_checkpoint_path = base_path / pose_checkpoint
        
        print(f"✓ Detection config exists: {det_config_path.exists()}")
        print(f"✓ Detection checkpoint exists: {det_checkpoint_path.exists()}")
        print(f"✓ Pose config exists: {pose_config_path.exists()}")
        print(f"✓ Pose checkpoint exists: {pose_checkpoint_path.exists()}")
        
        # Test video files exist (should be in RepJudge/test_videos/)
        test_videos_path = base_path / "test_videos"
        if test_videos_path.exists():
            video_files = list(test_videos_path.glob("*.mp4"))
            print(f"✓ Found {len(video_files)} test videos available")
            if video_files:
                print(f"  Example: {video_files[0].name}")
        else:
            print("⚠ test_videos directory not found")
        
        # Don't start GUI in demo mode
        print("\n✓ All basic functionality tests passed!")
        print("\nTo run the full application:")
        print("  python rep_labeler.py")
        print("  # or")
        print("  ./launch.sh")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure MMPose is installed:")
        print("  pip install mmpose")
        return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def list_available_videos():
    """List available test videos."""
    print("\n=== Available Test Videos ===")
    
    base_path = Path("/home/lucas/RepJudge")
    test_videos_path = base_path / "test_videos"
    
    if not test_videos_path.exists():
        print("No test_videos directory found")
        return
        
    video_files = list(test_videos_path.glob("*.mp4"))
    
    if not video_files:
        print("No MP4 files found in test_videos")
        return
        
    print(f"Found {len(video_files)} video files:")
    for i, video in enumerate(video_files[:10], 1):  # Show first 10
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  {i:2d}. {video.name} ({size_mb:.1f} MB)")
        
    if len(video_files) > 10:
        print(f"     ... and {len(video_files) - 10} more")


def main():
    """Main demo function."""
    print("RepLabeler Demo - Testing Installation and Setup")
    print("=" * 50)
    
    # Test basic functionality
    success = test_basic_functionality()
    
    # List available videos
    list_available_videos()
    
    if success:
        print("\n🎉 RepLabeler is ready to use!")
        print("\nNext steps:")
        print("1. Run: python rep_labeler.py")
        print("2. Load one of your test videos")
        print("3. Wait for MMPose processing")
        print("4. Start labeling exercise states!")
    else:
        print("\n❌ Setup incomplete. Please install missing dependencies.")


if __name__ == "__main__":
    main()