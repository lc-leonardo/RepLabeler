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

        ttk.Label(sidebar, textvariable=self.root_dir_var, wraplength=220).grid(
            row=1, column=0, sticky="ew"
        )

        ttk.Label(sidebar, text="Exercises", padding=(0, 10, 0, 0)).grid(
            row=2, column=0, sticky="w"
        )
        self.exercise_list = tk.Listbox(sidebar, exportselection=False, height=8)
        self.exercise_list.grid(row=3, column=0, sticky="nsew")
        self.exercise_list.bind("<<ListboxSelect>>", self.on_exercise_select)

        ttk.Label(sidebar, text="Samples", padding=(0, 10, 0, 0)).grid(
            row=4, column=0, sticky="w"
        )
        self.sample_list = tk.Listbox(sidebar, exportselection=False, height=10)
        self.sample_list.grid(row=5, column=0, sticky="nsew")
        self.sample_list.bind("<Double-Button-1>", self.on_sample_double_click)

        ttk.Button(sidebar, text="Load selected sample", command=self.load_selected_sample).grid(
            row=6, column=0, pady=(10, 0), sticky="ew"
        )

        sidebar.rowconfigure(3, weight=1)
        sidebar.rowconfigure(5, weight=2)

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

        ttk.Label(info_frame, textvariable=self.current_state_var, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Label(info_frame, textvariable=self.sequence_var, anchor="w", foreground="#444").grid(
            row=1, column=0, sticky="ew"
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
        try:
            labels = [item["label"] for item in annotations]
        except (TypeError, KeyError):
            return None
        if labels[0] != "prep" or labels[-1] != "finish":
            return None
        expected_middle = ["rep" if bit == "1" else "no-rep" for bit in self.binary_label]
        if labels[1:-1] != expected_middle:
            return None
        segments: List[Segment] = []
        for item in annotations[:-1]:  # exclude finish for editing
            try:
                start = int(item["start"])
                end = int(item["end"])
                label = str(item["label"])
            except (KeyError, TypeError, ValueError):
                return None
            segments.append(Segment(start, end, label))
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

    def save_annotations(self) -> None:
        if not self.sample_json_paths or not self.state_sequence:
            return
        required_segments = len(self.state_sequence) - 1
        if len(self.recorded_segments) != required_segments:
            messagebox.showwarning(
                "Incomplete",
                "Please mark every state (prep and each rep/no-rep) before saving.",
            )
            return
        if self.total_frames <= 0:
            messagebox.showerror("Video error", "Unable to determine frame count for the video.")
            return

        finish_start = min(self.recorded_segments[-1].end + 1, self.total_frames - 1)
        finish_segment = Segment(finish_start, self.total_frames - 1, "finish")
        segments_to_save = [seg.as_dict() for seg in self.recorded_segments]
        segments_to_save.append(finish_segment.as_dict())

        for json_path in self.sample_json_paths:
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except json.JSONDecodeError as exc:
                messagebox.showerror("JSON error", f"Failed to load {json_path.name}: {exc}")
                return
            data["binary_label"] = self.binary_label
            data["annotations"] = segments_to_save
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)

        self.status_var.set("Annotations saved to all JSON files")
        messagebox.showinfo("Saved", "Annotations have been written to all JSON files.")

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
        has_sample = bool(self.capture and self.state_sequence)
        mark_enabled = has_sample and self.current_state_index < len(self.state_sequence) - 1
        undo_enabled = has_sample and self.current_state_index > 0
        save_ready = has_sample and self.current_state_index == len(self.state_sequence) - 1

        self.play_button.configure(state=tk.NORMAL if has_sample else tk.DISABLED)
        self.mark_button.configure(state=tk.NORMAL if mark_enabled else tk.DISABLED)
        self.undo_button.configure(state=tk.NORMAL if undo_enabled else tk.DISABLED)
        self.clear_button.configure(state=tk.NORMAL if has_sample else tk.DISABLED)
        self.save_button.configure(state=tk.NORMAL if save_ready else tk.DISABLED)

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


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if sys.platform == "darwin":  # macOS nice default
        style.theme_use("clam")
    app = VideoPoseLabellerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
