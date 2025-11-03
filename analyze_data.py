#!/usr/bin/env python3
"""
Example script demonstrating how to load and analyze RepLabeler output data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import argparse


def load_annotation_data(json_file):
    """Load annotation data from RepLabeler JSON output."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def analyze_state_distribution(data):
    """Analyze the distribution of exercise states."""
    states = []
    for frame in data['frame_data']:
        if frame['state_label']:
            states.append(frame['state_label'])
    
    state_counts = Counter(states)
    total_frames = len(data['frame_data'])
    labeled_frames = len(states)
    
    print(f"=== State Distribution Analysis ===")
    print(f"Total frames: {total_frames}")
    print(f"Labeled frames: {labeled_frames}")
    print(f"Labeling completeness: {labeled_frames/total_frames*100:.1f}%\n")
    
    print("State distribution:")
    for state, count in state_counts.most_common():
        percentage = count / labeled_frames * 100
        print(f"  {state}: {count} frames ({percentage:.1f}%)")
    
    return state_counts


def plot_state_timeline(data, save_path=None):
    """Create a timeline plot of exercise states."""
    states = []
    frame_indices = []
    
    for frame in data['frame_data']:
        frame_indices.append(frame['frame_index'])
        states.append(frame['state_label'] if frame['state_label'] else 'Unlabeled')
    
    unique_states = list(set(states))
    state_to_num = {state: i for i, state in enumerate(unique_states)}
    
    state_numbers = [state_to_num[state] for state in states]
    
    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, state_numbers, marker='o', markersize=2)
    plt.yticks(range(len(unique_states)), unique_states)
    plt.xlabel('Frame Index')
    plt.ylabel('Exercise State')
    plt.title('Exercise State Timeline')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to: {save_path}")
    else:
        plt.show()


def analyze_keypoint_confidence(data):
    """Analyze keypoint detection confidence."""
    confidences = []
    
    for frame in data['frame_data']:
        if frame['keypoints']:
            for keypoint in frame['keypoints']:
                if len(keypoint) >= 3:  # [x, y, confidence]
                    confidences.append(keypoint[2])
    
    if not confidences:
        print("No keypoint confidence data found")
        return
    
    confidences = np.array(confidences)
    
    print(f"\n=== Keypoint Confidence Analysis ===")
    print(f"Total keypoints: {len(confidences)}")
    print(f"Mean confidence: {confidences.mean():.3f}")
    print(f"Std confidence: {confidences.std():.3f}")
    print(f"Min confidence: {confidences.min():.3f}")
    print(f"Max confidence: {confidences.max():.3f}")
    print(f"Keypoints with conf > 0.5: {(confidences > 0.5).sum()} ({(confidences > 0.5).mean()*100:.1f}%)")


def detect_repetitions(data):
    """Simple repetition detection based on state transitions."""
    states = []
    for frame in data['frame_data']:
        if frame['state_label']:
            states.append(frame['state_label'])
    
    if not states:
        print("No labeled states found for repetition detection")
        return
    
    # Look for common repetition patterns
    rep_patterns = [
        ["Start Position", "Concentric Phase", "Top Position", "Eccentric Phase"],
        ["Bottom Position", "Concentric Phase", "Top Position", "Eccentric Phase"],
    ]
    
    repetitions = []
    current_rep = []
    
    for i, state in enumerate(states):
        current_rep.append((i, state))
        
        # Check if we completed a repetition pattern
        current_states = [s for _, s in current_rep[-4:]]  # Last 4 states
        
        for pattern in rep_patterns:
            if len(current_states) >= len(pattern) and current_states[-len(pattern):] == pattern:
                repetitions.append(current_rep[-len(pattern):])
                current_rep = []
                break
    
    print(f"\n=== Repetition Analysis ===")
    print(f"Detected repetitions: {len(repetitions)}")
    
    if repetitions:
        rep_lengths = [len(rep) for rep in repetitions]
        print(f"Average repetition length: {np.mean(rep_lengths):.1f} frames")
        
        for i, rep in enumerate(repetitions[:5]):  # Show first 5 reps
            start_frame = rep[0][0]
            end_frame = rep[-1][0]
            print(f"  Rep {i+1}: frames {start_frame}-{end_frame} ({end_frame-start_frame+1} frames)")


def generate_summary_report(json_file, output_dir=None):
    """Generate a comprehensive summary report."""
    data = load_annotation_data(json_file)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = Path.cwd()
    
    print(f"=== RepLabeler Data Analysis Report ===")
    print(f"File: {json_file}")
    print(f"Video: {data['video_info']['path']}")
    print(f"Model used: {data['video_info']['model_used']}")
    print(f"Created: {data['video_info']['created_at']}")
    print()
    
    # Analyze state distribution
    state_counts = analyze_state_distribution(data)
    
    # Analyze keypoint confidence
    analyze_keypoint_confidence(data)
    
    # Detect repetitions
    detect_repetitions(data)
    
    # Generate timeline plot
    plot_path = output_dir / f"timeline_{Path(json_file).stem}.png"
    plot_state_timeline(data, save_path=plot_path)
    
    # Save analysis summary
    summary_path = output_dir / f"analysis_{Path(json_file).stem}.txt"
    
    # This is a simplified summary - you could expand it
    with open(summary_path, 'w') as f:
        f.write(f"RepLabeler Analysis Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Video: {data['video_info']['path']}\n")
        f.write(f"Total frames: {data['video_info']['total_frames']}\n")
        f.write(f"Model: {data['video_info']['model_used']}\n\n")
        
        f.write("State Distribution:\n")
        for state, count in state_counts.most_common():
            f.write(f"  {state}: {count} frames\n")
    
    print(f"\nAnalysis complete! Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RepLabeler annotation data")
    parser.add_argument("json_file", help="Path to RepLabeler JSON output file")
    parser.add_argument("--output-dir", "-o", help="Output directory for analysis files")
    parser.add_argument("--plot-only", action="store_true", help="Only generate timeline plot")
    
    args = parser.parse_args()
    
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        return
    
    try:
        if args.plot_only:
            data = load_annotation_data(args.json_file)
            plot_state_timeline(data)
        else:
            generate_summary_report(args.json_file, args.output_dir)
            
    except Exception as e:
        print(f"Error analyzing data: {e}")


if __name__ == "__main__":
    main()