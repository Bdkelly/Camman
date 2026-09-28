"""Concatenate annotation lists without silently merging equal frame names."""

import argparse
import json
from pathlib import Path


def concatenate_json_files(json_file_1_path, json_file_2_path, output_combined_path):
    combined = []
    for filename in (json_file_1_path, json_file_2_path):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {filename}")
        combined.extend(data)
    output = Path(output_combined_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Concatenate two annotation lists, preserving metadata"
    )
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    concatenate_json_files(args.first, args.second, args.output)


if __name__ == "__main__":
    main()
