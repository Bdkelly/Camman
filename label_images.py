"""Manually label soccer balls in Camman training images.

    python label_images.py data/game_annotated/frames

Drag around each ball. Save + next commits the boxes. No ball + next saves
an explicit negative. Arrow keys browse frames without adding a label.
Labels are saved after each decision in manual_labels.json beside frames/ (or inside any other image directory).
Pass --output to choose a different file or resume work on that file.
"""

import argparse
import json
import os
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


def load_labels(path):
    if not path.exists():
        return {}
    return {
        record["frame"]: record["boxes"]
        for record in json.loads(path.read_text(encoding="utf-8"))
    }


def save_labels(path, labels):
    records = [{"frame": name, "boxes": boxes} for name, boxes in sorted(labels.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(records, stream, indent=2)
            stream.write("\n")
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def image_box(start, end, display_size, image_size):
    """Map a drag on the zoomed canvas to a clipped source-pixel box."""
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    width, height = image_size
    display_width, display_height = display_size
    x_scale = width / display_width
    y_scale = height / display_height
    box = [
        round(min(max(x1, 0), display_width) * x_scale),
        round(min(max(y1, 0), display_height) * y_scale),
        round(min(max(x2, 0), display_width) * x_scale),
        round(min(max(y2, 0), display_height) * y_scale),
    ]
    return box if box[0] < box[2] and box[1] < box[3] else None


class Labeler:
    def __init__(self, root, images, output):
        self.root = root
        self.images = images
        self.output = output
        self.labels = load_labels(output)
        self.index = next(
            (i for i, path in enumerate(images) if path.name not in self.labels), 0,
        )
        self.image = None
        self.boxes = []
        self.anchor = None
        self.zoom = 1.0

        root.title("Camman Ball Labeler")
        root.geometry("1100x760")
        toolbar = ttk.Frame(root, padding=6)
        toolbar.pack(fill="x")
        for title, action in [
            ("Previous ←", lambda: self.navigate(-1)),
            ("Next →", lambda: self.navigate(1)),
            ("Undo box", self.undo),
            ("Clear boxes", self.clear),
            ("Zoom +", lambda: self.change_zoom(1.5)),
            ("Zoom −", lambda: self.change_zoom(1 / 1.5)),
            ("Save + next (S)", self.save_boxes),
            ("No ball + next (N)", self.save_empty),
        ]:
            ttk.Button(toolbar, text=title, command=action).pack(side="left", padx=2)

        self.status = ttk.Label(root, padding=6)
        self.status.pack(side="bottom", fill="x")
        view = ttk.Frame(root)
        view.pack(fill="both", expand=True)
        view.rowconfigure(0, weight=1)
        view.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(view, background="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        horizontal = ttk.Scrollbar(view, orient="horizontal", command=self.canvas.xview)
        vertical = ttk.Scrollbar(view, orient="vertical", command=self.canvas.yview)
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind("<ButtonPress-1>", self.start_box)
        self.canvas.bind("<B1-Motion>", self.drag_box)
        self.canvas.bind("<ButtonRelease-1>", self.finish_box)
        root.bind("<Left>", lambda event: self.navigate(-1))
        root.bind("<Right>", lambda event: self.navigate(1))
        root.bind("<Key-s>", lambda event: self.save_boxes())
        root.bind("<Key-n>", lambda event: self.save_empty())
        root.bind("<Control-z>", lambda event: self.undo())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.update_idletasks()
        self.open_frame(self.index)
        root.focus_set()

    @property
    def name(self):
        return self.images[self.index].name

    def open_frame(self, index):
        path = self.images[index]
        try:
            with Image.open(path) as image:
                self.image = image.convert("RGB")
        except OSError as exc:
            messagebox.showerror("Cannot open image", f"{path}\n{exc}", parent=self.root)
            return
        self.index = index
        self.boxes = list(self.labels.get(self.name, []))
        self.anchor = None
        self.zoom = 1.0
        self.render()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def render(self):
        width, height = self.image.size
        fit = min(self.canvas.winfo_width() / width, self.canvas.winfo_height() / height, 1.0)
        self.display_size = tuple(max(1, round(size * fit * self.zoom)) for size in (width, height))
        self.photo = ImageTk.PhotoImage(self.image.resize(self.display_size, Image.Resampling.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, *self.display_size))
        self.render_boxes()

    def render_boxes(self):
        self.canvas.delete("box")
        width, height = self.image.size
        dw, dh = self.display_size
        for x1, y1, x2, y2 in self.boxes:
            self.canvas.create_rectangle(
                x1 * dw / width, y1 * dh / height, x2 * dw / width, y2 * dh / height,
                outline="#47ed88", width=2, tags="box",
            )
        saved = self.labels.get(self.name)
        state = "Unlabeled"
        if saved is not None:
            state = "Saved" if saved else "Saved as no ball"
        if self.boxes != (saved or []):
            state = "Unsaved edits"
        self.status.configure(
            text=f"{self.index + 1}/{len(self.images)}  |  {self.name}  |  "
                 f"{len(self.boxes)} box(es)  |  {state}",
        )

    def change_zoom(self, factor):
        if self.image is not None:
            self.zoom = max(1.0, min(8.0, self.zoom * factor))
            self.render()

    def point(self, event):
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def start_box(self, event):
        if self.image is not None:
            self.anchor = self.point(event)

    def drag_box(self, event):
        if self.anchor is not None:
            self.canvas.delete("draft")
            self.canvas.create_rectangle(
                *self.anchor, *self.point(event), outline="#ffd166", width=2, tags="draft",
            )

    def finish_box(self, event):
        if self.anchor is None:
            return
        box = image_box(self.anchor, self.point(event), self.display_size, self.image.size)
        self.anchor = None
        self.canvas.delete("draft")
        if box is not None:
            self.boxes.append(box)
            self.render_boxes()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.render_boxes()

    def clear(self):
        if self.image is not None:
            self.boxes.clear()
            self.render_boxes()

    def save_boxes(self):
        if self.boxes:
            self.commit(self.boxes)
        elif self.image is not None:
            messagebox.showinfo(
                "No boxes drawn", "Use No ball + next to confirm an empty frame.",
                parent=self.root,
            )

    def save_empty(self):
        if self.image is None:
            return
        if self.boxes and not messagebox.askyesno(
            "Confirm no ball", "Remove these boxes and mark this image as no ball?",
            parent=self.root,
        ):
            return
        self.commit([])

    def commit(self, boxes):
        labels = {**self.labels, self.name: list(boxes)}
        try:
            save_labels(self.output, labels)
        except OSError as exc:
            messagebox.showerror("Could not save labels", str(exc), parent=self.root)
            return
        self.labels = labels
        self.boxes = list(boxes)
        self.render_boxes()
        if self.index + 1 < len(self.images):
            self.open_frame(self.index + 1)

    def can_leave(self):
        return self.boxes == self.labels.get(self.name, []) or messagebox.askyesno(
            "Unsaved edits", "Discard unsaved boxes for this image?", parent=self.root,
        )

    def navigate(self, offset):
        index = self.index + offset
        if 0 <= index < len(self.images) and self.can_leave():
            self.open_frame(index)

    def close(self):
        if self.can_leave():
            self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, help="Directory of clean video frames or other images")
    parser.add_argument("--output", type=Path, help="Manual labels JSON to create or resume")
    args = parser.parse_args(argv)
    directory = args.images.expanduser()
    if not directory.is_dir():
        parser.error(f"Image directory not found: {directory}")
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        parser.error(f"No JPG or PNG images found in {directory}")
    labels_dir = directory.parent if directory.name == "frames" else directory
    output = args.output.expanduser() if args.output else labels_dir / "manual_labels.json"
    if output.suffix.lower() != ".json":
        parser.error("--output must be a JSON file")
    try:
        root = tk.Tk()
        Labeler(root, images, output)
    except (OSError, ValueError, TypeError, KeyError, tk.TclError) as exc:
        parser.exit(1, f"Cannot start labeler: {exc}\n")
    root.mainloop()


if __name__ == "__main__":
    main()
