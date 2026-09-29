"""Run with python -m backend.tools.check_device on the target machine."""

import argparse
import platform

import cv2
import torch
import torchvision

from backend.capture import has_gstreamer
from backend.runtime import jetson_model, resolve_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check deployment packages and native vision ops")
    parser.add_argument("--device", help="Device for operator checks, e.g. cuda:0")
    parser.add_argument(
        "--verify-ops", action="store_true", help="Execute torchvision NMS and ROIAlign"
    )
    args = parser.parse_args(argv)
    print(f"Platform: {platform.machine()}; {jetson_model() or 'no Jetson device tree'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"CUDA build: {torch.version.cuda}; cuDNN: {torch.backends.cudnn.version()}")
    print(f"OpenCV: {cv2.__version__} ({cv2.__file__}); GStreamer: {has_gstreamer()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    if args.verify_ops:
        try:
            device = resolve_runtime(device=args.device).device
            with torch.inference_mode():
                boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]], device=device)
                scores = torch.tensor([0.9, 0.8], device=device)
                keep = torchvision.ops.nms(boxes, scores, 0.5)
                pooled = torchvision.ops.roi_align(
                    torch.ones(1, 1, 4, 4, device=device), [boxes], output_size=2
                )
                # Host copies force completion so broken CUDA ops fail this check.
                assert keep.cpu().numel() == 2 and pooled.cpu().shape == (2, 1, 2, 2)
            print(f"torchvision NMS and ROIAlign: OK on {device}")
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            parser.exit(1, f"Operator check failed: {exc}\n")


if __name__ == "__main__":
    main()
