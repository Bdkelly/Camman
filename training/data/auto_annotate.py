"""Generate candidate annotations with one model load for the entire image directory."""

import argparse
import json
from pathlib import Path

import cv2
import torch

from backend.detection import FrameTransform, get_ball_detection
from backend.models import load_model_from_path


def annotate(images, model_path, output, *, confidence=0.98, preview_dir=None, device=None):
    paths = sorted(
        path for path in Path(images).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ValueError(f"No images found in {images}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_from_path(model_path, device)
    transform = FrameTransform()
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read {path}")
        detections, annotated = get_ball_detection(
            model,
            frame,
            transform,
            device,
            confidence,
            max_detections=None,
            draw=preview_dir is not None,
        )
        boxes = [
            {
                "Label": "Ball",
                "x_min": d["box"][0],
                "y_min": d["box"][1],
                "x_max": d["box"][2],
                "y_max": d["box"][3],
            }
            for d in detections
        ]
        records.append({path.name: boxes})
        if preview_dir is not None:
            if not cv2.imwrite(str(preview_dir / f"inferred_{path.name}"), annotated):
                raise OSError(f"Could not save preview for {path}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate ball annotation candidates for manual review"
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview-dir", help="Save inferred_ image previews for the review tool")
    parser.add_argument("--confidence", type=float, default=0.98)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    records = annotate(
        args.images,
        args.model,
        args.output,
        confidence=args.confidence,
        preview_dir=args.preview_dir,
        device=args.device,
    )
    print(
        f"Saved candidates for {len(records)} frames to {args.output}. Review these before training."
    )


if __name__ == "__main__":
    main()
