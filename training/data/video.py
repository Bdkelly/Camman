"""Frame extraction and assembly; run with python -m training.data.video."""

import argparse
import subprocess
from pathlib import Path

import cv2


def extract_frames(video, output, fps=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-n", "-i", str(video)]
    if fps is not None:
        if fps <= 0:
            raise ValueError("FPS must be positive")
        command += ["-vf", f"fps={fps}"]
    command += ["-start_number", "0", str(output / "frame_%05d.jpg")]
    subprocess.run(command, check=True)


def create_video_from_images(image_folder, video_name, fps=30):
    if fps <= 0:
        raise ValueError("FPS must be positive")
    images = sorted(
        path
        for path in Path(image_folder).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise ValueError(f"No images in {image_folder}")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f"Cannot read {images[0]}")
    height, width = first.shape[:2]
    Path(video_name).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        if not writer.isOpened():
            raise OSError(f"Cannot open video writer: {video_name}")
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Unreadable image or inconsistent dimensions: {path}")
            writer.write(frame)
    finally:
        writer.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract video frames or assemble an MP4")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("video")
    extract.add_argument("output")
    extract.add_argument("--fps", type=float, help="Omit to keep all source frames")
    assemble = commands.add_parser("assemble")
    assemble.add_argument("images")
    assemble.add_argument("output")
    assemble.add_argument("--fps", type=float, default=30)
    args = parser.parse_args(argv)
    if args.command == "extract":
        extract_frames(args.video, args.output, args.fps)
    else:
        create_video_from_images(args.images, args.output, args.fps)


if __name__ == "__main__":
    main()
