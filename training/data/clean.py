"""Convert legacy inference/Label Studio annotations to one record per image."""

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

from PIL import Image


def load_annotations(json_path, images_dir):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Annotations must be a JSON list")
    sources = {source for item in data for source in item.get("videos", {})}
    if len(sources) > 1:
        raise ValueError(
            "Multiple video sources found. Clean each video separately to avoid frame-name collisions."
        )
    frames = OrderedDict()
    images_dir = Path(images_dir)

    def add(frame, box=None):
        frames.setdefault(str(frame), [])
        if box is not None and list(box) not in frames[str(frame)]:
            frames[str(frame)].append(list(box))

    def sequence(values):
        for record in values:
            if record.get("frame") is None or not record.get("enabled", True):
                continue
            name = f"frame_{int(record['frame']):05d}.jpg"
            if (
                not (images_dir / name).is_file()
                and (images_dir / name).with_suffix(".png").is_file()
            ):
                name = str(Path(name).with_suffix(".png"))
            with Image.open(images_dir / name) as image:
                width, height = image.size
            x, y, w, h = (float(record[k]) for k in ("x", "y", "width", "height"))
            add(
                name,
                [x * width / 100, y * height / 100, (x + w) * width / 100, (y + h) * height / 100],
            )

    for item in data:
        if "videos" in item:
            continue
        if "frame" in item:
            name = item["frame"]
            add(name)
            if "boxes" in item:
                for box in item["boxes"]:
                    add(name, box)
            else:
                add(name, [item[k] for k in ("x_min", "y_min", "x_max", "y_max")])
        elif "box" in item:
            for annotation in item["box"]:
                sequence(annotation.get("sequence", []))
        elif "annotations" in item:
            for annotation in item["annotations"]:
                for result in annotation.get("result", []):
                    sequence(result.get("value", {}).get("sequence", []))
        else:
            for name, boxes in item.items():
                if not isinstance(boxes, list):
                    raise ValueError(f"Unsupported annotation entry: {name}")
                add(name)
                for box in boxes:
                    add(name, [box[k] for k in ("x_min", "y_min", "x_max", "y_max")])
    if not frames:
        raise ValueError("No frame annotations found")
    records = []
    for name, boxes in frames.items():
        with Image.open(images_dir / name) as image:
            width, height = image.size
        checked = []
        for x1, y1, x2, y2 in boxes:
            coords = [
                max(0.0, float(x1)),
                max(0.0, float(y1)),
                min(float(width), float(x2)),
                min(float(height), float(y2)),
            ]
            if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
                raise ValueError(f"Non-finite box in {name}")
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError(f"Invalid box in {name}: {coords}")
            checked.append(coords)
        records.append({"frame": name, "boxes": checked})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Clean ball annotations (pixel xyxy boxes, including negative frames)"
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    records = load_annotations(args.annotations, args.images)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} frames to {output}")


if __name__ == "__main__":
    main()
