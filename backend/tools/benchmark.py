"""Measure real batch-one detector latency on target hardware, without Qt or motors."""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from backend.capture import add_capture_arguments, open_capture, parse_source
from backend.detection import FrameTransform, get_ball_detection
from backend.models import load_model_from_path
from backend.runtime import add_runtime_arguments, jetson_model, resolve_runtime


def summarize_ms(samples):
    values = np.asarray(samples) * 1000
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def measure(model, cap, runtime, *, frames=100, warmup=5):
    transform = FrameTransform()
    detection_times, capture_times, total_times = [], [], []
    count = detected_frames = 0
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    while count < warmup + frames:
        iteration_started = time.perf_counter()
        ok, frame = cap.read()
        captured = time.perf_counter()
        if not ok:
            break
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        started = time.perf_counter()
        boxes, _ = get_ball_detection(
            model, frame, transform, runtime.device, draw=False, precision=runtime.precision
        )
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        ended = time.perf_counter()
        if count >= warmup:
            detection_times.append(ended - started)
            capture_times.append(captured - iteration_started)
            total_times.append(ended - iteration_started)
            detected_frames += bool(boxes)
        count += 1
    if not detection_times:
        raise ValueError("No measured frames after warmup. Use a longer source or reduce --warmup")
    return {
        "measured_frames": len(detection_times),
        "requested_frames": frames,
        "warmup_frames": warmup,
        "frames_with_detection": detected_frames,
        "detection_ms": summarize_ms(detection_times),
        "capture_ms": summarize_ms(capture_times),
        "sequential_throughput_fps": len(total_times) / sum(total_times),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(runtime.device) / 1024**2
            if runtime.device.type == "cuda"
            else None
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Camman detector .pth checkpoint")
    add_capture_arguments(parser, playback=False)
    add_runtime_arguments(parser)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args(argv)
    cap = None
    try:
        if args.frames < 1 or args.warmup < 0:
            raise ValueError("Frames must be positive and warmup must be nonnegative")
        for value in (args.capture_width, args.capture_height, args.capture_fps):
            if value is not None and not value > 0:
                raise ValueError("Capture dimensions and FPS must be positive")
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
        # Measure every sequential frame at maximum throughput, without application
        # rate caps or live-frame dropping. Capture time is reported separately.
        cap = open_capture(
            parse_source(args.source),
            backend=args.capture_backend,
            width=args.capture_width,
            height=args.capture_height,
            fps=args.capture_fps,
        )
        report = {
            "profile": runtime.profile,
            "device": str(runtime.device),
            "precision": runtime.precision,
            "detector_size": runtime.detector_size,
            "preprocessing_size": 640,
            "cpu_threads": torch.get_num_threads(),
            "opencv_threads": cv2.getNumThreads(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "opencv": cv2.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(runtime.device)
            if runtime.device.type == "cuda"
            else None,
            "platform": platform.platform(),
            "jetson_model": jetson_model(),
            "model": str(Path(args.model).resolve()),
            "capture_backend": cap.getBackendName(),
            "capture_width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "capture_height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "confidence_threshold": 0.98,
            "note": "Unpaced sequential capture + detection; excludes Qt, drawing and serial. "
            "Detection includes preprocessing/transfers/postprocessing. "
            "Detection count is not an accuracy metric. CUDA memory excludes driver/video/Qt.",
            **measure(model, cap, runtime, frames=args.frames, warmup=args.warmup),
        }
        result = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result + "\n")
        print(result)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
