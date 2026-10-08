"""Cache video detections once; policy training then uses small numeric trajectories."""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend.control import box_center
from backend.detection import FrameTransform, get_ball_detection
from backend.models import load_model_from_path
from backend.runtime import add_runtime_arguments, resolve_runtime


@dataclass
class TrackSequence:
    centers: np.ndarray
    detected: np.ndarray
    fps: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.centers = np.asarray(self.centers, dtype=np.float32)
        self.detected = np.asarray(self.detected, dtype=bool)
        self.fps = float(self.fps)
        if self.centers.ndim != 2 or self.centers.shape[1] != 2 or len(self.centers) < 2:
            raise ValueError("Tracks need at least two [x,y] centers")
        if self.detected.shape != (len(self.centers),) or not np.isfinite(self.centers).all():
            raise ValueError("Track masks/coordinates are invalid")
        if np.any((self.centers < 0) | (self.centers > 1)) or not 0 < self.fps <= 1000:
            raise ValueError("Track centers must be normalized 0..1 and FPS must be positive")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                version=1,
                centers=self.centers,
                detected=self.detected,
                fps=self.fps,
                metadata=json.dumps(self.metadata),
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            if int(values["version"]) != 1:
                raise ValueError("Unsupported trajectory format")
            return cls(
                values["centers"],
                values["detected"],
                float(values["fps"]),
                json.loads(str(values["metadata"])),
            )

    def slice(self, start, stop):
        return TrackSequence(
            self.centers[start:stop].copy(),
            self.detected[start:stop].copy(),
            self.fps,
            {**self.metadata, "slice": [start, stop]},
        )

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(self.centers.tobytes())
        digest.update(self.detected.tobytes())
        digest.update(str(self.fps).encode())
        return digest.hexdigest()


def extract_tracks(
    video,
    model,
    device,
    *,
    confidence=0.98,
    precision="fp32",
    max_frames=None,
    fps_override=None,
    progress=print,
):
    if not 0 <= confidence <= 1 or (max_frames is not None and max_frames < 2):
        raise ValueError("Invalid confidence threshold or frame limit")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open training video: {video}")
        fps = fps_override if fps_override is not None else cap.get(cv2.CAP_PROP_FPS)
        if not 0 < fps <= 1000:
            raise ValueError("Video FPS is invalid; provide --fps explicitly")
        transform = FrameTransform()
        centers, detected = [], []
        while max_frames is None or len(centers) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, _ = get_ball_detection(
                model, frame, transform, device, confidence, precision=precision, draw=False
            )
            center = box_center(boxes, frame.shape[1], frame.shape[0])
            centers.append(center if center is not None else (0.5, 0.5))
            detected.append(center is not None)
            if progress and len(centers) % 300 == 0:
                progress(f"Cached {len(centers)} frames, {sum(detected)} detections")
        return TrackSequence(
            centers,
            detected,
            fps,
            {
                "kind": "video_detections",
                "source": str(Path(video).resolve()),
                "confidence": confidence,
                "precision": precision,
            },
        )
    finally:
        cap.release()


def synthetic_tracks(frames=3000, fps=30.0, seed=0):
    if frames < 2 or not 0 < fps <= 1000:
        raise ValueError("Synthetic tracks require at least two frames and valid FPS")
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / fps
    phase = rng.uniform(-np.pi, np.pi)
    centers = np.column_stack((0.5 + 0.3 * np.sin(t * 0.25 + phase), 0.5 + 0.1 * np.sin(t * 0.1)))
    detected = np.ones(frames, dtype=bool)
    detected[120::251] = False
    return TrackSequence(centers, detected, fps, {"kind": "synthetic", "seed": seed})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument(
        "--synthetic", action="store_true", help="Software demo only; no game training data"
    )
    parser.add_argument("--model", help="Required with --video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.98)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--fps", type=float, help="Override missing video FPS; synthetic default: 30"
    )
    parser.add_argument("--seed", type=int, default=0)
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    if args.synthetic:
        tracks = synthetic_tracks(
            args.max_frames if args.max_frames is not None else 3000,
            args.fps if args.fps is not None else 30,
            args.seed,
        )
    else:
        if not args.model:
            parser.error("--model is required with --video")
        runtime = resolve_runtime(
            profile=args.profile,
            device=args.device,
            precision=args.precision,
            detector_size=args.detector_size,
            cpu_threads=args.cpu_threads,
        )
        runtime.apply()
        model = load_model_from_path(
            args.model, runtime.device, detector_size=runtime.detector_size
        )
        tracks = extract_tracks(
            args.video,
            model,
            runtime.device,
            confidence=args.confidence,
            precision=runtime.precision,
            max_frames=args.max_frames,
            fps_override=args.fps,
        )
        tracks.metadata.update(
            model=str(Path(args.model).resolve()), detector_size=runtime.detector_size
        )
    tracks.save(args.output)
    print(
        f"Saved {len(tracks.centers)} frames ({int(tracks.detected.sum())} detections): {args.output}"
    )


if __name__ == "__main__":
    main()
