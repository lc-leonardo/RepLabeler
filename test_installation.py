#!/usr/bin/env python3
"""
Test script to verify RepLabeler installation and dependencies
"""

import sys
import importlib
from pathlib import Path


def test_import(module_name, required=True):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        status = "✗" if required else "⚠"
        print(f"{status} {module_name} - {e}")
        return False


def test_python_version():
    """Test Python version compatibility."""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def test_file_structure():
    """Test if required files exist."""
    required_files = [
        "rep_labeler.py",
        "README.md",
        "requirements.txt",
        "config.json",
        "launch.sh",
        "analyze_data.py"
    ]
    
    base_path = Path(__file__).parent
    all_exist = True
    
    print("\nFile structure:")
    for file in required_files:
        file_path = base_path / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")
            all_exist = False
    
    return all_exist


def test_mmpose_model():
    """Test MMPose model availability."""
    try:
        from mmpose.apis.inferencers import MMPoseInferencer
        model_name = "cspnext-m_udp_8xb64-210e_crowdpose-256x192"
        
        print(f"\nTesting MMPose model: {model_name}")
        
        # Try to initialize the model (this will download if not available)
        inferencer = MMPoseInferencer(model_name)
        print("✓ MMPose model loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ MMPose model test failed: {e}")
        print("  Note: Model will be downloaded on first use")
        return False


def main():
    """Run all tests."""
    print("=== RepLabeler Installation Test ===\n")
    
    all_passed = True
    
    # Test Python version
    print("Python version:")
    all_passed &= test_python_version()
    
    # Test core dependencies
    print("\nCore dependencies:")
    core_deps = [
        ("tkinter", True),
        ("PIL", True),  # Pillow
        ("cv2", True),  # opencv-python
        ("numpy", True),
    ]
    
    for dep, required in core_deps:
        all_passed &= test_import(dep, required)
    
    # Test MMPose dependencies
    print("\nMMPose dependencies:")
    mmpose_deps = [
        ("mmpose", True),
        ("mmcv", True),
        ("mmengine", True),
        ("torch", True),
        ("torchvision", True),
    ]
    
    for dep, required in mmpose_deps:
        all_passed &= test_import(dep, required)
    
    # Test optional dependencies
    print("\nOptional dependencies:")
    optional_deps = [
        ("matplotlib", False),
        ("scipy", False),
    ]
    
    for dep, required in optional_deps:
        test_import(dep, required)
    
    # Test file structure
    all_passed &= test_file_structure()
    
    # Test MMPose model (optional, may take time)
    print("\nMMPose model test (optional):")
    response = input("Test MMPose model loading? This may download ~100MB (y/N): ")
    if response.lower() in ['y', 'yes']:
        test_mmpose_model()
    else:
        print("Skipped MMPose model test")
    
    # Summary
    print("\n" + "="*50)
    if all_passed:
        print("✓ All critical tests passed! RepLabeler should work correctly.")
        print("\nTo start the application:")
        print("  cd RepLabeler")
        print("  python rep_labeler.py")
        print("  # or")
        print("  ./launch.sh")
    else:
        print("✗ Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nFor MMPose installation, see:")
        print("  https://mmpose.readthedocs.io/en/latest/installation.html")


if __name__ == "__main__":
    main()