"""Add video-source metadata to a legacy annotation list using explicit paths."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write a copy of legacy annotations with source metadata"
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video", required=True, help="Video name")
    parser.add_argument("--images", required=True)
    parser.add_argument("--format", default="mp4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected an annotation list")
    entry = {"videos": {args.video: {"format": args.format, "path": args.images}}}
    if data and "videos" in data[0] and args.video in data[0]["videos"]:
        data[0]["videos"][args.video] = entry["videos"][args.video]
    else:
        data.insert(0, entry)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
