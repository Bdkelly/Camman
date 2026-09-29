"""Inference-only settings. Training never applies these process-wide tunings."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch


def jetson_model():
    try:
        return Path("/proc/device-tree/model").read_text().rstrip("\x00\n")
    except OSError:
        return ""


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str
    device: torch.device
    precision: str
    detector_size: int
    preview_fps: float
    inference_fps: float
    cpu_threads: int | None

    def apply(self):
        if self.cpu_threads is not None:
            torch.set_num_threads(self.cpu_threads)
            cv2.setNumThreads(1)

    def describe(self):
        return (
            f"Profile: {self.profile}; device: {self.device}; {self.precision}; "
            f"detector: {self.detector_size}px; preview cap: {self.preview_fps:g} FPS; "
            f"inference cap: {self.inference_fps:g} FPS (0 = unlimited)"
        )


def resolve_runtime(
    *,
    profile="auto",
    device=None,
    precision="auto",
    detector_size=None,
    preview_fps=None,
    inference_fps=None,
    cpu_threads=None,
):
    if profile not in {"auto", "standard", "jetson"}:
        raise ValueError("Profile must be auto, standard or jetson")
    if profile == "auto":
        profile = "jetson" if "jetson" in jetson_model().lower() else "standard"
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Check the JetPack/PyTorch build or use --device cpu")
    if precision == "auto":
        precision = "fp16" if profile == "jetson" and device.type == "cuda" else "fp32"
    if precision not in {"fp16", "fp32"}:
        raise ValueError("Precision must be auto, fp16 or fp32")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 inference requires CUDA; use --precision fp32 on CPU")
    if detector_size is None:
        detector_size = 640 if profile == "jetson" else 800
    if detector_size < 128 or detector_size > 1536 or detector_size % 32:
        raise ValueError("Detector size must be a multiple of 32 between 128 and 1536")
    if preview_fps is None:
        preview_fps = 15.0 if profile == "jetson" else 30.0
    if inference_fps is None:
        inference_fps = 15.0 if profile == "jetson" else 0.0
    if not 1 <= preview_fps <= 120 or not 0 <= inference_fps <= 120:
        raise ValueError("Preview FPS must be 1..120 and inference FPS must be 0..120")
    if cpu_threads is None and profile == "jetson":
        cpu_threads = 2
    if cpu_threads is not None and cpu_threads < 1:
        raise ValueError("CPU threads must be at least 1")
    return RuntimeConfig(
        profile, device, precision, detector_size, preview_fps, inference_fps, cpu_threads
    )


def add_runtime_arguments(parser):
    parser.add_argument(
        "--profile",
        choices=("auto", "standard", "jetson"),
        default="auto",
        help="auto detects Jetson; standard preserves the 800px FP32 detector",
    )
    parser.add_argument("--device", help="Torch device; default: CUDA if available, otherwise CPU")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto")
    parser.add_argument("--detector-size", type=int, help="Internal detector size, e.g. 640 or 800")
    parser.add_argument("--cpu-threads", type=int, help="Torch CPU threads (Jetson default: 2)")
