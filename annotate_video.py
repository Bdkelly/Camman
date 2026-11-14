"""Cut and annotate a video for Camman detector training.

Run from the Camman repository root:
    python annotate_video.py
    python annotate_video.py "game.mp4" --model detector.pth --start 30 --end 60

Start and end are seconds on the source's nominal FPS timeline (end exclusive).
The runner saves each extracted frame as PNG, then runs ball detection.
Coordinates in predictions.json and candidates.json are source pixels in
[x_min, y_min, x_max, y_max] order. The review MP4 has a fixed frame rate and no audio. Review candidates.json
against the original frames before using it with camman-train --annotations
and --images. Only detections become candidates; confirm negatives separately.
"""

import argparse
import json
import math
import shutil
from pathlib import Path


def annotate_video(video, model_path, output, *, start, end, confidence, device):
    import cv2
    import torch

    from backend.detection import FrameTransform, get_ball_detection
    from backend.models import load_model_from_path

    capture = cv2.VideoCapture(str(video))
    writer = None
    created_output = False
    completed = False
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot determine the video frame rate: {video}")

        first = math.ceil(start * fps)
        stop = math.ceil(end * fps) if end is not None else None
        if stop is not None and stop <= first:
            raise ValueError("The selected interval contains no frames")
        # Advance sequentially for consistent source-frame numbering across codecs.
        for _ in range(first):
            if not capture.grab():
                raise ValueError("The start time is past the last video frame")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("No readable frames in the selected interval")

        output.mkdir(parents=True, exist_ok=False)
        created_output = True
        frames_dir = output / "frames"
        frames_dir.mkdir()
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = load_model_from_path(model_path, device)
        transform = FrameTransform()
        height, width = frame.shape[:2]
        # MP4 encoders can truncate odd dimensions. Pad only the review video.
        pad_bottom, pad_right = height % 2, width % 2
        writer = cv2.VideoWriter(
            str(output / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width + pad_right, height + pad_bottom),
        )
        if not writer.isOpened():
            raise OSError(f"Cannot create MP4 in {output}")

        count = 0
        candidates = []
        predictions = []
        while ok:
            source_index = first + count
            name = f"frame_{source_index:09d}.png"
            if not cv2.imwrite(str(frames_dir / name), frame):
                raise OSError(f"Cannot write training image: {name}")
            # PNG preserves the decoded pixels used for inference and training.
            detections, preview = get_ball_detection(
                model, frame, transform, device, confidence, max_detections=1,
            )
            if pad_bottom or pad_right:
                preview = cv2.copyMakeBorder(
                    preview, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT,
                )
            writer.write(preview)
            prediction = {
                "frame": name,
                "clip_frame": count,
                "source_frame": source_index,
                "source_seconds": source_index / fps,
                "status": "no_detection",
                "ball": None,
            }
            if detections:
                detection = detections[0]
                x1, y1, x2, y2 = detection["box"]
                box = [x1, y1, x2, y2]
                prediction["status"] = "detected"
                prediction["ball"] = {
                    "box": box,
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    "confidence": detection["score"],
                }
                candidates.append({"frame": name, "boxes": [box]})
            predictions.append(prediction)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} frames on {device}", flush=True)
            if stop is not None and first + count >= stop:
                break
            ok, frame = capture.read()

        writer.release()
        writer = None
        if (output / "annotated.mp4").stat().st_size == 0:
            raise OSError(f"MP4 encoder produced an empty file in {output}")
        (output / "predictions.json").write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8",
        )
        (output / "candidates.json").write_text(
            json.dumps(candidates, indent=2) + "\n", encoding="utf-8",
        )
        completed = True
        return count, len(candidates)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if created_output and not completed:
            shutil.rmtree(output)


def prompt_path(label):
    return Path(input(f"{label}: ").strip().strip('"').strip("'")).expanduser()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, help="Prompt for the path if omitted")
    parser.add_argument("--model", type=Path, help="Detector .pth checkpoint")
    parser.add_argument("--output", type=Path, help="New output folder (default: data/<video>_annotated)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds (default: end of video)")
    parser.add_argument("--confidence", type=float, default=0.98, help="Minimum detection score")
    parser.add_argument("--device", help="Override automatic CUDA/CPU selection")
    args = parser.parse_args(argv)
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    if args.end is not None and (not math.isfinite(args.end) or args.end <= args.start):
        parser.error("--end must be finite and greater than --start")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")

    from backend.config import models_directory

    try:
        video = args.video.expanduser() if args.video else prompt_path("Video file path")
        if not video.is_file():
            parser.error(f"Video file not found: {video}")
        model = args.model
        if model is None:
            installed = sorted(path for path in models_directory().glob("*.pth") if path.is_file())
            if len(installed) == 1:
                model = installed[0]
            else:
                model = prompt_path("Detector checkpoint (.pth) path")
        model = model.expanduser()
        if not model.is_file():
            parser.error(f"Detector checkpoint not found: {model}")
        output = (args.output or (Path("data") / f"{video.stem}_annotated")).expanduser()
        print(f"Detector: {model}", flush=True)
        count, detected = annotate_video(
            video, model, output, start=args.start, end=args.end,
            confidence=args.confidence, device=args.device,
        )
    except (EOFError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Annotation failed: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nAnnotation cancelled.\n")
    print(f"Saved {count} frames and {detected} candidate labels to {output}")
    print("Review the candidate labels before training.")


if __name__ == "__main__":
    main()
