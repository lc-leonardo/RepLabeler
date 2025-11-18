#!/usr/bin/env python3
"""Video Pose Repetition Labeller using Tkinter and OpenCV."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "OpenCV (cv2) is required. Install it with 'pip install opencv-python'."
    ) from exc

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Pillow is required. Install it with 'pip install Pillow'."
    ) from exc


@dataclass
class Segment:
    """Represents a labeled segment."""

    start: int
    end: int
    label: str

    def as_dict(self) -> dict:
        return {"start": int(self.start), "end": int(self.end), "label": self.label}


class VideoPoseLabellerApp:
    """Main Tkinter application for labeling repetition segments."""

    MAX_DISPLAY_WIDTH = 960
    MAX_DISPLAY_HEIGHT = 540

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Pose Repetition Labeller")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Path management
        self.json_root: Optional[Path] = None
        self.dataset_root: Optional[Path] = None
        self.sample_json_paths: List[Path] = []
        self.binary_label: str = ""
        self.state_sequence: List[str] = []
        self.new_video_mode: bool = False  # Track if we're working with a new unlabeled video
        self.new_video_path: Optional[Path] = None

        # Video playback state
        self.capture: Optional[cv2.VideoCapture] = None
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.frame_delay_ms: int = 33
        self.playing: bool = False
        self.after_id: Optional[str] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.slider_updating: bool = False

        # Annotation state
        self.recorded_segments: List[Segment] = []
        self.current_state_index: int = 0
        self.state_start_frame: int = 0

        # UI state variables
        self.root_dir_var = tk.StringVar(value="Choose a json_keypoints root folder")
        self.current_state_var = tk.StringVar(value="No sample loaded")
        self.sequence_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Select a folder to begin")
        self.frame_info_var = tk.StringVar(value="Frame: - / -")

        # Build the UI widgets
        self._build_ui()

        # Pre-select default json root if it exists
        default_root = Path.cwd() / "CFRep" / "json_keypoints"
        if default_root.exists():
            self.set_json_root(default_root)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, padding=10)
        sidebar.grid(row=0, column=0, sticky="ns")

        ttk.Button(
            sidebar,
            text="Select json_keypoints folder…",
            command=self.choose_json_root,
        ).grid(row=0, column=0, pady=(0, 6), sticky="ew")
        
        ttk.Button(
            sidebar,
            text="Load New Video…",
            command=self.load_new_video,
        ).grid(row=1, column=0, pady=(0, 6), sticky="ew")

        ttk.Label(sidebar, textvariable=self.root_dir_var, wraplength=220).grid(
            row=2, column=0, sticky="ew"
        )

        ttk.Label(sidebar, text="Exercises", padding=(0, 10, 0, 0)).grid(
            row=3, column=0, sticky="w"
        )
        self.exercise_list = tk.Listbox(sidebar, exportselection=False, height=8)
        self.exercise_list.grid(row=4, column=0, sticky="nsew")
        self.exercise_list.bind("<<ListboxSelect>>", self.on_exercise_select)

        ttk.Label(sidebar, text="Samples", padding=(0, 10, 0, 0)).grid(
            row=5, column=0, sticky="w"
        )
        self.sample_list = tk.Listbox(sidebar, exportselection=False, height=10)
        self.sample_list.grid(row=6, column=0, sticky="nsew")
        self.sample_list.bind("<Double-Button-1>", self.on_sample_double_click)

        ttk.Button(sidebar, text="Load selected sample", command=self.load_selected_sample).grid(
            row=7, column=0, pady=(10, 0), sticky="ew"
        )

        # Build aggregated video_config.json button
        ttk.Button(
            sidebar,
            text="Build video_config.json",
            command=self.build_video_config,
        ).grid(row=8, column=0, pady=(6, 0), sticky="ew")

        sidebar.rowconfigure(4, weight=1)
        sidebar.rowconfigure(6, weight=2)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self.video_label = ttk.Label(main, anchor="center", relief=tk.SUNKEN)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(main)
        controls.grid(row=1, column=0, sticky="ew", pady=8)
        controls.columnconfigure(4, weight=1)

        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=2)

        ttk.Button(controls, text="⟨ Frame", command=lambda: self.step_frame(-1)).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(controls, text="Frame ⟩", command=lambda: self.step_frame(1)).grid(
            row=0, column=2, padx=2
        )

        self.mark_button = ttk.Button(
            controls, text="Mark end of current state", command=self.mark_current_state
        )
        self.mark_button.grid(row=0, column=3, padx=8)
        
        # Manual windowing buttons for new video mode
        self.mark_prep_button = ttk.Button(
            controls, text="Mark as prep", command=lambda: self.mark_manual_segment("prep")
        )
        
        self.mark_rep_button = ttk.Button(
            controls, text="Mark as rep", command=lambda: self.mark_manual_segment("rep")
        )
        
        self.mark_norep_button = ttk.Button(
            controls, text="Mark as no-rep", command=lambda: self.mark_manual_segment("no-rep")
        )
        
        self.mark_finish_button = ttk.Button(
            controls, text="Mark as finish", command=lambda: self.mark_manual_segment("finish")
        )

        self.undo_button = ttk.Button(
            controls, text="Undo last mark", command=self.undo_last_mark
        )
        self.undo_button.grid(row=0, column=4, padx=2, sticky="w")

        self.clear_button = ttk.Button(
            controls, text="Clear annotations", command=self.clear_annotations
        )
        self.clear_button.grid(row=0, column=5, padx=2)

        self.save_button = ttk.Button(controls, text="Save annotations", command=self.save_annotations)
        self.save_button.grid(row=0, column=6, padx=2)

        slider_frame = ttk.Frame(main)
        slider_frame.grid(row=2, column=0, sticky="ew")
        slider_frame.columnconfigure(0, weight=1)

        self.frame_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            command=self.on_slider_moved,
        )
        self.frame_slider.grid(row=0, column=0, sticky="ew")

        ttk.Label(slider_frame, textvariable=self.frame_info_var, width=20, anchor="e").grid(
            row=0, column=1, padx=(8, 0)
        )

        info_frame = ttk.Frame(main)
        info_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        info_frame.columnconfigure(0, weight=1)

        # Binary label editor
        binary_frame = ttk.Frame(info_frame)
        binary_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        binary_frame.columnconfigure(1, weight=1)
        
        ttk.Label(binary_frame, text="Binary Label:").grid(row=0, column=0, sticky="w")
        self.binary_entry = ttk.Entry(binary_frame, width=20)
        self.binary_entry.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.binary_entry.bind("<Return>", self.on_binary_label_changed)
        
        ttk.Button(binary_frame, text="Update", command=self.on_binary_label_changed).grid(
            row=0, column=2, padx=(5, 0)
        )

        ttk.Label(info_frame, textvariable=self.current_state_var, anchor="w").grid(
            row=1, column=0, sticky="ew"
        )
        ttk.Label(info_frame, textvariable=self.sequence_var, anchor="w", foreground="#444").grid(
            row=2, column=0, sticky="ew"
        )

        self.annotation_tree = ttk.Treeview(
            main,
            columns=("start", "end", "label"),
            show="headings",
            height=6,
        )
        self.annotation_tree.heading("start", text="Start")
        self.annotation_tree.heading("end", text="End")
        self.annotation_tree.heading("label", text="Label")
        self.annotation_tree.column("start", width=80, anchor="center")
        self.annotation_tree.column("end", width=80, anchor="center")
        self.annotation_tree.column("label", width=120, anchor="center")
        self.annotation_tree.grid(row=4, column=0, sticky="nsew", pady=(6, 0))
        
        # Add double-click editing for annotations
        self.annotation_tree.bind("<Double-1>", self.on_annotation_double_click)
        
        # Add buttons for annotation management
        annotation_buttons = ttk.Frame(main)
        annotation_buttons.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        
        ttk.Button(annotation_buttons, text="Edit Selected", command=self.edit_selected_annotation).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(annotation_buttons, text="Delete Selected", command=self.delete_selected_annotation).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(annotation_buttons, text="Insert Segment", command=self.insert_segment).grid(
            row=0, column=2, padx=2
        )

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=6)
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

        self._update_buttons()

    # ------------------------------------------------------------------
    # Folder and sample selection
    # ------------------------------------------------------------------
    def choose_json_root(self) -> None:
        selected = filedialog.askdirectory(title="Select json_keypoints folder")
        if not selected:
            return
        self.set_json_root(Path(selected))

    def set_json_root(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if not path.exists() or not path.is_dir():
            messagebox.showerror("Invalid folder", f"{path} is not a valid directory")
            return

        self.json_root = path
        self.dataset_root = path.parent
        self.root_dir_var.set(str(path))
        self.status_var.set("Pick an exercise to continue")
        self.populate_exercises()

    def populate_exercises(self) -> None:
        self.exercise_list.delete(0, tk.END)
        self.sample_list.delete(0, tk.END)
        if not self.json_root:
            return
        for entry in sorted(self.json_root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                self.exercise_list.insert(tk.END, entry.name)

    def on_exercise_select(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        del event
        if not self.json_root:
            return
        try:
            selection = self.exercise_list.curselection()
            if not selection:
                return
            exercise = self.exercise_list.get(selection[0])
        except tk.TclError:
            return
        self.populate_samples(exercise)

    def populate_samples(self, exercise: str) -> None:
        self.sample_list.delete(0, tk.END)
        if not self.json_root:
            return
        exercise_dir = self.json_root / exercise
        if not exercise_dir.exists():
            return
        for entry in sorted(exercise_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                self.sample_list.insert(tk.END, entry.name)

    def on_sample_double_click(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        del event
        self.load_selected_sample()

    def load_selected_sample(self) -> None:
        if not self.json_root:
            messagebox.showwarning("Select folder", "Please choose a json_keypoints folder first")
            return
        try:
            exercise_idx = self.exercise_list.curselection()
            if not exercise_idx:
                messagebox.showinfo("Select exercise", "Please select an exercise")
                return
            exercise = self.exercise_list.get(exercise_idx[0])
            sample_idx = self.sample_list.curselection()
            if not sample_idx:
                messagebox.showinfo("Select sample", "Please select a sample")
                return
            sample = self.sample_list.get(sample_idx[0])
        except tk.TclError:
            return

        self.load_sample(exercise, sample)

    # ------------------------------------------------------------------
    # Sample loading and validation
    # ------------------------------------------------------------------
    def load_sample(self, exercise: str, sample: str) -> None:
        assert self.json_root is not None
        self.pause_video()
        self.close_video()

        sample_dir = self.json_root / exercise / sample
        json_files = sorted(sample_dir.glob("*.json"))
        if not json_files:
            messagebox.showerror("Missing JSON", "No JSON files found for the selected sample")
            return

        primary_json_path = json_files[0]
        try:
            with primary_json_path.open("r", encoding="utf-8") as handle:
                primary_data = json.load(handle)
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid JSON", f"Failed to parse {primary_json_path.name}: {exc}")
            return

        binary_label = primary_data.get("binary_label", "")
        if not binary_label:
            binary_label = self.prompt_binary_label(sample)
            if binary_label is None:
                self.status_var.set("Binary label required to proceed")
                return

        if not all(ch in "01" for ch in binary_label):
            messagebox.showerror("Invalid binary label", "Binary label must contain only 0 and 1")
            return

        self.binary_label = binary_label
        self.state_sequence = self._build_state_sequence(binary_label)
        self.sample_json_paths = json_files

        video_path = self._resolve_video_path(primary_data, sample)
        if video_path is None or not video_path.exists():
            messagebox.showerror("Video not found", "Unable to locate the source video for this sample")
            return

        if not self._open_video(video_path):
            return

        self.recorded_segments = []
        existing_segments = self._validate_existing_annotations(primary_data)
        if existing_segments:
            if messagebox.askyesno(
                "Existing annotations",
                "Existing annotations were found. Do you want to load them?"
            ):
                self._apply_existing_segments(existing_segments)
            else:
                self.recorded_segments = []

        self.current_state_index = len(self.recorded_segments)
        self.state_start_frame = 0 if not self.recorded_segments else self.recorded_segments[-1].end + 1
        self.state_start_frame = min(self.state_start_frame, max(self.total_frames - 1, 0))
        self.seek_to_frame(0)
        self.status_var.set(f"Loaded {exercise} / {sample}")
        self._refresh_state_ui()
        self._update_annotation_view()
        self._update_buttons()
        
        # Update the binary label entry field
        self.binary_entry.delete(0, tk.END)
        self.binary_entry.insert(0, self.binary_label)

    # ------------------------------------------------------------------
    # Binary label and annotation editing
    # ------------------------------------------------------------------
    def on_binary_label_changed(self, event=None) -> None:
        """Handle changes to the binary label field."""
        new_binary = self.binary_entry.get().strip()
        
        if not all(ch in "01" for ch in new_binary):
            messagebox.showerror("Invalid binary label", "Binary label must contain only 0 and 1")
            self.binary_entry.delete(0, tk.END)
            self.binary_entry.insert(0, self.binary_label)
            return
        
        if new_binary != self.binary_label:
            if self.recorded_segments:
                response = messagebox.askyesno(
                    "Binary label changed",
                    "Changing the binary label will clear existing annotations. Continue?"
                )
                if not response:
                    self.binary_entry.delete(0, tk.END)
                    self.binary_entry.insert(0, self.binary_label)
                    return
                
            self.binary_label = new_binary
            self.state_sequence = self._build_state_sequence(new_binary)
            self.recorded_segments.clear()
            self.current_state_index = 0
            self.state_start_frame = 0
            self.seek_to_frame(0)
            self._refresh_state_ui()
            self._update_annotation_view()
            self._update_buttons()
    
    def on_annotation_double_click(self, event) -> None:
        """Handle double-click on annotation tree items."""
        selection = self.annotation_tree.selection()
        if selection:
            self.edit_selected_annotation()
    
    def edit_selected_annotation(self) -> None:
        """Edit the selected annotation segment."""
        selection = self.annotation_tree.selection()
        if not selection:
            messagebox.showinfo("No selection", "Please select an annotation to edit.")
            return
            
        item = selection[0]
        values = self.annotation_tree.item(item, "values")
        if not values or len(values) < 3:
            return
            
        start_frame = int(values[0])
        end_frame = int(values[1])
        label = values[2]
        
        # Don't allow editing finish segment
        if label == "finish":
            messagebox.showinfo("Cannot edit", "The 'finish' segment is automatically managed.")
            return
        
        # Find the segment index
        segment_index = None
        for i, seg in enumerate(self.recorded_segments):
            if seg.start == start_frame and seg.end == end_frame and seg.label == label:
                segment_index = i
                break
                
        if segment_index is None:
            messagebox.showerror("Error", "Could not find the selected segment.")
            return
            
        # Create edit dialog
        dialog = SegmentEditDialog(self.root, start_frame, end_frame, label, self.total_frames)
        if dialog.result:
            new_start, new_end, new_label = dialog.result
            
            # Update the segment
            self.recorded_segments[segment_index].start = new_start
            self.recorded_segments[segment_index].end = new_end
            self.recorded_segments[segment_index].label = new_label
            
            # Re-sort segments and update UI
            self.recorded_segments.sort(key=lambda seg: seg.start)
            self._update_annotation_view()
            self.status_var.set("Annotation updated")
    
    def delete_selected_annotation(self) -> None:
        """Delete the selected annotation segment."""
        selection = self.annotation_tree.selection()
        if not selection:
            messagebox.showinfo("No selection", "Please select an annotation to delete.")
            return
            
        item = selection[0]
        values = self.annotation_tree.item(item, "values")
        if not values or len(values) < 3:
            return
            
        start_frame = int(values[0])
        end_frame = int(values[1])
        label = values[2]
        
        # Don't allow deleting finish segment
        if label == "finish":
            messagebox.showinfo("Cannot delete", "The 'finish' segment cannot be deleted.")
            return
        
        response = messagebox.askyesno(
            "Delete segment", 
            f"Delete segment '{label}' ({start_frame}-{end_frame})?"
        )
        if not response:
            return
        
        # Find and remove the segment
        for i, seg in enumerate(self.recorded_segments):
            if seg.start == start_frame and seg.end == end_frame and seg.label == label:
                self.recorded_segments.pop(i)
                break
                
        # Update state tracking
        self.current_state_index = len(self.recorded_segments)
        self.state_start_frame = 0 if not self.recorded_segments else self.recorded_segments[-1].end + 1
        self.state_start_frame = min(self.state_start_frame, max(self.total_frames - 1, 0))
        
        self._refresh_state_ui()
        self._update_annotation_view()
        self._update_buttons()
        self.status_var.set("Annotation deleted")
    
    def insert_segment(self) -> None:
        """Insert a new segment at the current frame."""
        if not self.capture or not self.state_sequence:
            messagebox.showwarning("No video", "Please load a video first.")
            return
            
        # Create dialog for new segment
        dialog = SegmentEditDialog(
            self.root, 
            self.current_frame, 
            min(self.current_frame + 30, self.total_frames - 1),
            "rep", 
            self.total_frames
        )
        if dialog.result:
            new_start, new_end, new_label = dialog.result
            
            # Insert the new segment
            new_segment = Segment(new_start, new_end, new_label)
            self.recorded_segments.append(new_segment)
            
            # Re-sort segments and update UI
            self.recorded_segments.sort(key=lambda seg: seg.start)
            self._update_annotation_view()
            self.status_var.set("Segment inserted")

    # ------------------------------------------------------------------
    # Build aggregated video_config.json
    # ------------------------------------------------------------------
    def build_video_config(self) -> None:
        """Scan json_keypoints and build/update CFRep/CFRep/video_config.json.

        Rules:
        - Only include videos that have annotations.
        - Include full segments list (including 'finish' if present).
        - Exclude derived rep_windows to keep schema minimal.
        - Compute binary_label from JSON or derive it from the segments if missing.
        - rep_count is number of segments labeled 'rep'.
        - Incremental: if video_config.json exists, update/append entries by filename.
        Schema per entry:
            {
                filename, exercise, binary_label, rep_count, segments: [ {start,end,label}, ... ]
            }
        """
        if not self.json_root:
            messagebox.showwarning("Select folder", "Please choose a json_keypoints folder first")
            return

        output_path = self.json_root.parent / "video_config.json"

        # Load existing aggregated file if present
        existing_by_filename: dict[str, dict] = {}
        if output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as f:
                    existing_list = json.load(f)
                if isinstance(existing_list, list):
                    for item in existing_list:
                        fn = item.get("filename")
                        if fn:
                            existing_by_filename[str(fn)] = item
            except Exception:
                # If unreadable, start fresh
                existing_by_filename = {}

        processed = 0
        skipped = 0

        # Traverse exercises and samples
        try:
            exercise_dirs = [d for d in sorted(self.json_root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        except FileNotFoundError:
            messagebox.showerror("Folder not found", f"Cannot access {self.json_root}")
            return

        for exercise_dir in exercise_dirs:
            for sample_dir in sorted(exercise_dir.iterdir()):
                if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                    continue
                json_files = sorted(sample_dir.glob("*.json"))
                if not json_files:
                    continue

                # Use the first JSON in the folder as primary
                primary = json_files[0]
                try:
                    with primary.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    skipped += 1
                    continue

                annotations = data.get("annotations")
                if not isinstance(annotations, list) or not annotations:
                    # Skip videos without annotations
                    skipped += 1
                    continue

                # Normalize segments and build rep_windows
                segments: list[dict] = []
                try:
                    for seg in annotations:
                        start = int(seg["start"])  # type: ignore[index]
                        end = int(seg["end"])      # type: ignore[index]
                        label = str(seg["label"])  # type: ignore[index]
                        segments.append({"start": start, "end": end, "label": label})
                except Exception:
                    skipped += 1
                    continue

                # Count reps (segments labeled 'rep')
                rep_count = sum(1 for s in segments if s.get("label") == "rep")

                # Determine filename and exercise
                video_path_val = data.get("video_path")
                if video_path_val:
                    filename = Path(str(video_path_val)).name
                else:
                    # Fallback to folder name assumption
                    filename = f"{sample_dir.name}.mp4"

                # Determine binary_label (prefer existing in JSON, else derive)
                binary_label = str(data.get("binary_label") or "")
                if not binary_label:
                    # Derive from segments by mapping middle labels
                    labels_in_order = [s.get("label") for s in segments]
                    # remove prep and finish if present
                    middle = [lbl for lbl in labels_in_order if lbl in ("rep", "no-rep")]
                    try:
                        binary_label = "".join("1" if lbl == "rep" else "0" for lbl in middle)
                    except Exception:
                        binary_label = ""

                entry = {
                    "filename": filename,
                    "exercise": exercise_dir.name,
                    "binary_label": binary_label,
                    "rep_count": rep_count,
                    "segments": segments,
                }

                existing_by_filename[filename] = entry
                processed += 1

        # Write back as a list sorted by filename
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            aggregated_list = [existing_by_filename[k] for k in sorted(existing_by_filename.keys())]
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(aggregated_list, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Write error", f"Failed to write {output_path}: {exc}")
            return

        self.status_var.set(
            f"Updated video_config.json — processed {processed}, skipped {skipped}"
        )
        messagebox.showinfo(
            "Compilation complete",
            f"video_config.json written at:\n{output_path}\n\nProcessed: {processed}\nSkipped (no annotations): {skipped}",
        )

    def prompt_binary_label(self, sample: str) -> Optional[str]:
        return simpledialog.askstring(
            "Binary label missing",
            f"Enter the binary label (e.g. 1010) for sample '{sample}':",
            parent=self.root,
        )

    def _build_state_sequence(self, binary_label: str) -> List[str]:
        sequence = ["prep"]
        sequence.extend("rep" if bit == "1" else "no-rep" for bit in binary_label)
        sequence.append("finish")
        return sequence

    def _resolve_video_path(self, primary_data: dict, sample: str) -> Optional[Path]:
        if self.dataset_root is None:
            return None
        candidate = self.dataset_root / f"{sample}.mp4"
        if candidate.exists():
            return candidate
        video_path_str = primary_data.get("video_path")
        if video_path_str:
            path_candidate = Path(video_path_str)
            if path_candidate.is_absolute():
                return path_candidate
            candidates = [self.dataset_root / video_path_str, self.dataset_root.parent / video_path_str]
            for option in candidates:
                if option.exists():
                    return option
        return None

    def _open_video(self, video_path: Path) -> bool:
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            messagebox.showerror("Video error", f"Could not open video: {video_path}")
            self.capture = None
            return False
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 1e-3:
            self.fps = 30.0
        self.frame_delay_ms = max(15, int(1000 / self.fps))
        self.current_frame = 0
        self.frame_slider.configure(from_=0, to=max(self.total_frames - 1, 1))
        return True

    def _validate_existing_annotations(self, primary_data: dict) -> Optional[List[Segment]]:
        annotations = primary_data.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            return None
        
        segments: List[Segment] = []
        try:
            for item in annotations:
                if item.get("label") == "finish":
                    continue  # Skip finish segment for editing
                    
                start = int(item["start"])
                end = int(item["end"])
                label = str(item["label"])
                
                # Basic validation
                if start < 0 or end < start:
                    return None
                    
                segments.append(Segment(start, end, label))
                
        except (KeyError, TypeError, ValueError):
            return None
            
        return segments
        return segments

    def _apply_existing_segments(self, segments: List[Segment]) -> None:
        self.recorded_segments = [Segment(seg.start, seg.end, seg.label) for seg in segments]
        # Ensure segments are sorted and clamped
        self.recorded_segments.sort(key=lambda seg: seg.start)
        if self.total_frames:
            for seg in self.recorded_segments:
                seg.start = max(0, min(seg.start, self.total_frames - 1))
                seg.end = max(0, min(seg.end, self.total_frames - 1))

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------
    def toggle_play(self) -> None:
        if not self.capture:
            return
        if self.playing:
            self.pause_video()
        else:
            self.playing = True
            self.play_button.configure(text="Pause")
            self._play_loop()

    def pause_video(self) -> None:
        if self.playing:
            self.playing = False
            self.play_button.configure(text="Play")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _play_loop(self) -> None:
        if not self.playing or not self.capture:
            return
        if self.current_frame >= self.total_frames - 1:
            self.pause_video()
            return
        self.current_frame += 1
        self.show_frame(self.current_frame)
        self.after_id = self.root.after(self.frame_delay_ms, self._play_loop)

    def step_frame(self, delta: int) -> None:
        if not self.capture or self.total_frames <= 0:
            return
        self.pause_video()
        new_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        self.seek_to_frame(new_frame)

    def seek_to_frame(self, frame_index: int) -> None:
        if not self.capture:
            return
        self.current_frame = max(0, min(self.total_frames - 1, frame_index))
        self.show_frame(self.current_frame)

    def on_slider_moved(self, value: str) -> None:  # pragma: no cover - UI callback
        if self.slider_updating:
            return
        try:
            frame_index = int(float(value))
        except ValueError:
            return
        self.pause_video()
        self.seek_to_frame(frame_index)

    def show_frame(self, frame_index: int) -> None:
        if not self.capture:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((self.MAX_DISPLAY_WIDTH, self.MAX_DISPLAY_HEIGHT), Image.LANCZOS)
        self.display_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.display_image)

        self.slider_updating = True
        self.frame_slider.set(frame_index)
        self.slider_updating = False

        total = max(self.total_frames - 1, 0)
        self.frame_info_var.set(f"Frame: {frame_index} / {total}")

    # ------------------------------------------------------------------
    # Annotation workflow
    # ------------------------------------------------------------------
    def mark_current_state(self) -> None:
        if not self.capture or not self.state_sequence:
            return
        if self.current_state_index >= len(self.state_sequence) - 1:
            messagebox.showinfo("All states marked", "All states have already been marked.")
            return
        self.pause_video()
        start_frame = 0 if self.current_state_index == 0 else self.state_start_frame
        end_frame = min(self.current_frame, self.total_frames - 1)
        if end_frame < start_frame:
            end_frame = start_frame
        label = self.state_sequence[self.current_state_index]

        # Overwrite any existing annotations beyond the current index
        self.recorded_segments = self.recorded_segments[: self.current_state_index]
        self.recorded_segments.append(Segment(start_frame, end_frame, label))
        self.current_state_index += 1
        self.state_start_frame = min(end_frame + 1, max(self.total_frames - 1, 0))
        if self.current_state_index < len(self.state_sequence) - 1:
            self.seek_to_frame(self.state_start_frame)
        self._refresh_state_ui()
        self._update_annotation_view()
        self._update_buttons()

    def undo_last_mark(self) -> None:
        if self.current_state_index == 0:
            return
        self.pause_video()
        removed = self.recorded_segments.pop()
        self.current_state_index -= 1
        self.state_start_frame = removed.start
        self.seek_to_frame(self.state_start_frame)
        self._refresh_state_ui()
        self._update_annotation_view()
        self._update_buttons()

    def clear_annotations(self) -> None:
        if not self.state_sequence:
            return
        if not messagebox.askyesno("Clear annotations", "Discard all marks for this sample?"):
            return
        self.pause_video()
        self.recorded_segments.clear()
        self.current_state_index = 0
        self.state_start_frame = 0
        self.seek_to_frame(0)
        self._refresh_state_ui()
        self._update_annotation_view()
        self._update_buttons()

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------
    def _refresh_state_ui(self) -> None:
        if not self.state_sequence:
            self.current_state_var.set("No sample loaded")
            self.sequence_var.set("")
            return
        if self.current_state_index < len(self.state_sequence) - 1:
            current_label = self.state_sequence[self.current_state_index]
            self.current_state_var.set(f"Current state: {current_label} — mark its end")
        else:
            self.current_state_var.set("All states marked. Finish will snap to the last frame on save.")

        decorated = []
        for idx, label in enumerate(self.state_sequence):
            if idx < self.current_state_index:
                decorated.append(f"✓ {label}")
            elif idx == self.current_state_index:
                decorated.append(f"→ {label}")
            else:
                decorated.append(label)
        self.sequence_var.set(" | ".join(decorated))

    def _update_annotation_view(self) -> None:
        for child in self.annotation_tree.get_children():
            self.annotation_tree.delete(child)
        for seg in self.recorded_segments:
            self.annotation_tree.insert("", tk.END, values=(seg.start, seg.end, seg.label))
        if self.state_sequence and len(self.recorded_segments) == len(self.state_sequence) - 1:
            finish_start = min(self.recorded_segments[-1].end + 1, self.total_frames - 1)
            self.annotation_tree.insert("", tk.END, values=(finish_start, self.total_frames - 1, "finish"))

    def _update_buttons(self) -> None:
        has_video = self.capture is not None
        can_mark = has_video and self.current_state_index < len(self.state_sequence)
        has_segments = bool(self.recorded_segments)
        can_save = has_video and self.recorded_segments and (self.current_state_index >= len(self.state_sequence) or self.new_video_mode)

        self.play_button.config(state="normal" if has_video else "disabled")
        
        # Show appropriate marking buttons based on mode
        if self.new_video_mode and not self.binary_label:
            # Manual windowing mode - show segment type buttons
            self.mark_button.grid_remove()
            self.mark_prep_button.grid(row=0, column=3, padx=2)
            self.mark_rep_button.grid(row=0, column=4, padx=2)
            self.mark_norep_button.grid(row=0, column=5, padx=2)
            self.mark_finish_button.grid(row=0, column=6, padx=2)
            
            # Adjust other button positions
            self.undo_button.grid(row=0, column=7, padx=2, sticky="w")
            self.clear_button.grid(row=0, column=8, padx=2)
            self.save_button.grid(row=0, column=9, padx=2)
        else:
            # Normal mode or binary-first mode - show state progression button
            self.mark_prep_button.grid_remove()
            self.mark_rep_button.grid_remove()
            self.mark_norep_button.grid_remove()
            self.mark_finish_button.grid_remove()
            
            self.mark_button.grid(row=0, column=3, padx=8)
            self.undo_button.grid(row=0, column=4, padx=2, sticky="w")
            self.clear_button.grid(row=0, column=5, padx=2)
            self.save_button.grid(row=0, column=6, padx=2)
        
        self.mark_button.config(state="normal" if can_mark else "disabled")
        self.mark_prep_button.config(state="normal" if has_video else "disabled")
        self.mark_rep_button.config(state="normal" if has_video else "disabled")
        self.mark_norep_button.config(state="normal" if has_video else "disabled")
        self.mark_finish_button.config(state="normal" if has_video else "disabled")
        
        self.undo_button.config(state="normal" if has_segments else "disabled")
        self.clear_button.config(state="normal" if has_segments else "disabled")
        self.save_button.config(state="normal" if can_save else "disabled")

    # ------------------------------------------------------------------
    # New video loading functionality
    # ------------------------------------------------------------------
    def load_new_video(self) -> None:
        """Load a new unlabeled video for annotation."""
        video_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        if not video_path:
            return
            
        video_path = Path(video_path)
        if not video_path.exists():
            messagebox.showerror("Video not found", f"Video file not found: {video_path}")
            return
            
        self.pause_video()
        self.close_video()
        
        if not self._open_video(video_path):
            messagebox.showerror("Video load error", "Failed to open video")
            return
            
        # Set up new video mode
        self.new_video_mode = True
        self.new_video_path = video_path
        self.binary_label = ""
        self.state_sequence = []
        self.recorded_segments = []
        self.sample_json_paths = []
        
        self.current_state_index = 0
        self.state_start_frame = 0
        self.seek_to_frame(0)
        
        self.status_var.set(f"Loaded new video: {video_path.name}")
        self._refresh_state_ui()
        self._update_annotation_view()
        self._update_buttons()
        
        # Clear binary entry for manual mode
        self.binary_entry.delete(0, tk.END)
        
        messagebox.showinfo(
            "New video loaded",
            "You can now either:\n\n"
            "1. Enter a binary label (0s and 1s) and follow the normal workflow, OR\n"
            "2. Use the manual windowing buttons (Mark as prep/rep/no-rep/finish) to create segments directly"
        )
    
    def mark_manual_segment(self, label: str) -> None:
        """Mark a segment with the specified label in manual mode."""
        if not self.new_video_mode:
            return
            
        if not self.capture:
            messagebox.showwarning("No video", "Please load a video first.")
            return
            
        # Determine start frame
        start_frame = 0 if not self.recorded_segments else self.recorded_segments[-1].end + 1
        end_frame = self.current_frame
        
        if end_frame <= start_frame:
            messagebox.showwarning("Invalid segment", "End frame must be after start frame.")
            return
            
        # Create and add the segment
        new_segment = Segment(start_frame, end_frame, label)
        self.recorded_segments.append(new_segment)
        
        self._update_annotation_view()
        self._update_buttons()
        self.status_var.set(f"Added {label} segment: {start_frame}-{end_frame}")
        
        # If this is finish, prepare for saving
        if label == "finish":
            self._derive_binary_from_segments()
    
    def _derive_binary_from_segments(self) -> None:
        """Derive binary label from manually created segments."""
        if not self.recorded_segments:
            return
            
        # Extract rep and no-rep segments (excluding prep and finish)
        middle_segments = [seg for seg in self.recorded_segments 
                          if seg.label in ("rep", "no-rep")]
        middle_segments.sort(key=lambda s: s.start)
        
        # Create binary string
        binary_label = "".join("1" if seg.label == "rep" else "0" 
                              for seg in middle_segments)
        
        self.binary_label = binary_label
        self.binary_entry.delete(0, tk.END)
        self.binary_entry.insert(0, binary_label)
        
        self.status_var.set(f"Derived binary label: {binary_label}")
    
    def save_annotations(self) -> None:
        """Save annotations - handles both existing and new video modes."""
        if not self.capture or not self.recorded_segments:
            messagebox.showwarning("Nothing to save", "No annotations to save.")
            return
            
        if self.new_video_mode:
            self._save_new_video_annotations()
        else:
            self._save_existing_annotations()
    
    def _save_new_video_annotations(self) -> None:
        """Save annotations for a new video."""
        if not self.new_video_path or not self.binary_label:
            messagebox.showerror("Missing data", "Binary label is required for saving.")
            return
            
        # Get video naming information
        naming_dialog = VideoNamingDialog(self.root)
        if not naming_dialog.result:
            return
            
        exercise, person, angle = naming_dialog.result
        new_video_name = f"{exercise}_{person}_{angle}.mp4"
        
        # Set up paths
        if self.json_root:
            dataset_root = self.json_root.parent
        else:
            # Prompt for dataset location
            dataset_root = filedialog.askdirectory(title="Select dataset root (CFRep/CFRep)")
            if not dataset_root:
                return
            dataset_root = Path(dataset_root)
            
        video_dest = dataset_root / new_video_name
        json_dir = dataset_root / "json_keypoints" / exercise / f"{exercise}_{person}_{angle}"
        
        try:
            # Create directory structure
            json_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy and rename video
            import shutil
            shutil.copy2(self.new_video_path, video_dest)
            
            # Create minimal JSON file
            json_data = {
                "video_path": f"CFRep/CFRep/{new_video_name}",
                "dataset_type": "CocoDataset",
                "exercise_type": exercise.upper().replace("_", "_"),
                "binary_label": self.binary_label,
                "frames": [],  # Empty frames for new videos
                "annotations": [seg.as_dict() for seg in self.recorded_segments]
            }
            
            # Save JSON file
            json_path = json_dir / f"{exercise}_{person}_{angle}_minimal.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
                
            # Update video_config.json
            self._update_video_configs(new_video_name, exercise, json_data)
            
            messagebox.showinfo(
                "Save successful",
                f"Video saved as: {new_video_name}\n"
                f"JSON created: {json_path.name}\n"
                f"Video configs updated."
            )
            
            # Clean up
            self.new_video_mode = False
            self.new_video_path = None
            
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save: {e}")
    
    def _save_existing_annotations(self) -> None:
        """Save annotations for existing videos (original functionality)."""
        if not self.sample_json_paths:
            messagebox.showerror("No files", "No JSON files loaded to save to.")
            return
            
        annotations = [seg.as_dict() for seg in self.recorded_segments]
        
        success_count = 0
        for json_path in self.sample_json_paths:
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                data["annotations"] = annotations
                data["binary_label"] = self.binary_label
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                success_count += 1
            except Exception:
                continue
                
        if success_count > 0:
            self.status_var.set(f"Annotations saved to {success_count} files")
            messagebox.showinfo("Save complete", f"Saved annotations to {success_count} JSON files")
        else:
            messagebox.showerror("Save failed", "Failed to save annotations to any files")
    
    def _update_video_configs(self, filename: str, exercise: str, json_data: dict) -> None:
        """Update both video_config.json and video_config.csv with new entry."""
        if not self.json_root:
            return
            
        config_dir = self.json_root.parent
        
        # Update video_config.json
        json_config_path = config_dir / "video_config.json"
        try:
            if json_config_path.exists():
                with json_config_path.open("r", encoding="utf-8") as f:
                    config_list = json.load(f)
            else:
                config_list = []
                
            # Create new entry
            segments = json_data.get("annotations", [])
            rep_count = sum(1 for s in segments if s.get("label") == "rep")
            
            new_entry = {
                "filename": filename,
                "exercise": exercise,
                "binary_label": json_data.get("binary_label", ""),
                "rep_count": rep_count,
                "segments": segments
            }
            
            config_list.append(new_entry)
            config_list.sort(key=lambda x: x.get("filename", ""))
            
            with json_config_path.open("w", encoding="utf-8") as f:
                json.dump(config_list, f, indent=2)
                
        except Exception as e:
            print(f"Failed to update video_config.json: {e}")
            
        # Update video_config.csv
        csv_config_path = config_dir / "video_config.csv"
        try:
            rep_count = sum(1 for s in json_data.get("annotations", []) if s.get("label") == "rep")
            csv_line = f"{filename},{exercise},{rep_count},{json_data.get('binary_label', '')}\n"
            
            with csv_config_path.open("a", encoding="utf-8") as f:
                f.write(csv_line)
                
        except Exception as e:
            print(f"Failed to update video_config.csv: {e}")

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------
    def close_video(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.current_frame = 0
        self.total_frames = 0
        self.display_image = None
        self.video_label.configure(image="")
        self.frame_info_var.set("Frame: - / -")

    def on_close(self) -> None:
        self.pause_video()
        self.close_video()
        self.root.destroy()


class SegmentEditDialog:
    """Dialog for editing segment start/end frames and labels."""
    
    def __init__(self, parent, start_frame: int, end_frame: int, label: str, max_frames: int):
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Edit Segment")
        self.dialog.geometry("300x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Start frame
        ttk.Label(main_frame, text="Start Frame:").grid(row=0, column=0, sticky="w", pady=2)
        self.start_var = tk.StringVar(value=str(start_frame))
        start_entry = ttk.Entry(main_frame, textvariable=self.start_var, width=10)
        start_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=2)
        
        # End frame
        ttk.Label(main_frame, text="End Frame:").grid(row=1, column=0, sticky="w", pady=2)
        self.end_var = tk.StringVar(value=str(end_frame))
        end_entry = ttk.Entry(main_frame, textvariable=self.end_var, width=10)
        end_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=2)
        
        # Label
        ttk.Label(main_frame, text="Label:").grid(row=2, column=0, sticky="w", pady=2)
        self.label_var = tk.StringVar(value=label)
        label_combo = ttk.Combobox(main_frame, textvariable=self.label_var, width=10)
        label_combo['values'] = ('prep', 'rep', 'no-rep', 'finish')
        label_combo.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=2)
        
        # Info label
        info_label = ttk.Label(main_frame, text=f"Max frame: {max_frames - 1}", foreground="gray")
        info_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(20, 0))
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).grid(row=0, column=1)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)
        
        # Store max frames for validation
        self.max_frames = max_frames
        
        # Focus on start entry
        start_entry.focus()
        start_entry.select_range(0, tk.END)
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def ok_clicked(self):
        """Validate and save the segment data."""
        try:
            start_frame = int(self.start_var.get())
            end_frame = int(self.end_var.get())
            label = self.label_var.get().strip()
            
            # Validation
            if start_frame < 0 or start_frame >= self.max_frames:
                messagebox.showerror("Invalid input", f"Start frame must be between 0 and {self.max_frames - 1}")
                return
            
            if end_frame < 0 or end_frame >= self.max_frames:
                messagebox.showerror("Invalid input", f"End frame must be between 0 and {self.max_frames - 1}")
                return
                
            if start_frame >= end_frame:
                messagebox.showerror("Invalid input", "Start frame must be less than end frame")
                return
            
            if not label or label not in ('prep', 'rep', 'no-rep', 'finish'):
                messagebox.showerror("Invalid input", "Please select a valid label")
                return
            
            self.result = (start_frame, end_frame, label)
            self.dialog.destroy()
            
        except ValueError:
            messagebox.showerror("Invalid input", "Frame numbers must be integers")
    
    def cancel_clicked(self):
        """Cancel the edit."""
        self.dialog.destroy()


class VideoNamingDialog:
    """Dialog for collecting video naming information for new videos."""
    
    def __init__(self, parent):
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Video Naming")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Instructions
        ttk.Label(main_frame, text="Enter video naming information:", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 15)
        )
        
        # Exercise type
        ttk.Label(main_frame, text="Exercise type:").grid(row=1, column=0, sticky="w", pady=5)
        self.exercise_var = tk.StringVar()
        exercise_combo = ttk.Combobox(main_frame, textvariable=self.exercise_var, width=20)
        exercise_combo['values'] = ('double_unders', 'push_ups', 'pull_ups', 'squats', 'burpees', 'other')
        exercise_combo.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)
        
        # Person identifier  
        ttk.Label(main_frame, text="Person identifier:").grid(row=2, column=0, sticky="w", pady=5)
        self.person_var = tk.StringVar()
        person_entry = ttk.Entry(main_frame, textvariable=self.person_var, width=20)
        person_entry.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=5)
        
        # Angle/Camera identifier
        ttk.Label(main_frame, text="Camera angle/ID:").grid(row=3, column=0, sticky="w", pady=5)
        self.angle_var = tk.StringVar()
        angle_entry = ttk.Entry(main_frame, textvariable=self.angle_var, width=20)
        angle_entry.grid(row=3, column=1, sticky="ew", padx=(10, 0), pady=5)
        
        # Example
        example_frame = ttk.LabelFrame(main_frame, text="Example", padding=10)
        example_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(20, 10))
        
        ttk.Label(example_frame, text="Exercise: double_unders", foreground="gray").grid(row=0, column=0, sticky="w")
        ttk.Label(example_frame, text="Person: diag_m2", foreground="gray").grid(row=1, column=0, sticky="w") 
        ttk.Label(example_frame, text="Angle: 9_7", foreground="gray").grid(row=2, column=0, sticky="w")
        ttk.Label(example_frame, text="Result: double_unders_diag_m2_9_7.mp4", foreground="blue").grid(row=3, column=0, sticky="w", pady=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(20, 0))
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).grid(row=0, column=1)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)
        
        # Focus on exercise combo
        exercise_combo.focus()
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def ok_clicked(self):
        """Validate and save the naming data."""
        exercise = self.exercise_var.get().strip()
        person = self.person_var.get().strip()
        angle = self.angle_var.get().strip()
        
        if not exercise:
            messagebox.showerror("Missing input", "Please enter an exercise type")
            return
            
        if not person:
            messagebox.showerror("Missing input", "Please enter a person identifier")
            return
            
        if not angle:
            messagebox.showerror("Missing input", "Please enter a camera angle/ID")
            return
            
        # Basic validation for safe filename
        for field, name in [(exercise, "exercise"), (person, "person"), (angle, "angle")]:
            if not all(c.isalnum() or c in "_-" for c in field):
                messagebox.showerror("Invalid input", f"{name} must contain only letters, numbers, underscores, and hyphens")
                return
        
        self.result = (exercise, person, angle)
        self.dialog.destroy()
    
    def cancel_clicked(self):
        """Cancel the naming."""
        self.dialog.destroy()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if sys.platform == "darwin":  # macOS nice default
        style.theme_use("clam")
    app = VideoPoseLabellerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
