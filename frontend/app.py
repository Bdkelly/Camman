"""Desktop entry point. Parse arguments before importing optional Qt dependencies."""

import argparse
import sys

from backend.capture import add_capture_arguments, parse_source
from backend.runtime import add_runtime_arguments, resolve_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description="Camman ball tracking application")
    add_capture_arguments(parser)
    add_runtime_arguments(parser)
    parser.add_argument("--model", help="Detector .pth checkpoint")
    parser.add_argument("--actor", help="Optional camera-control actor .pth checkpoint")
    parser.add_argument("--serial-port", help="ESP32 port, e.g. COM3, /dev/ttyUSB0, or auto")
    parser.add_argument(
        "--invert-pan", action="store_true", help="Reverse motor direction on the wire"
    )
    parser.add_argument(
        "--preview-fps", type=float, help="Preview ceiling (Jetson: 15; standard: 30)"
    )
    parser.add_argument("--inference-fps", type=float, help="Inference ceiling; 0 is unlimited")
    args = parser.parse_args(argv)
    try:
        runtime = resolve_runtime(
            profile=args.profile,
            device=args.device,
            precision=args.precision,
            detector_size=args.detector_size,
            cpu_threads=args.cpu_threads,
            preview_fps=args.preview_fps,
            inference_fps=args.inference_fps,
        )
        for value in (args.capture_width, args.capture_height, args.capture_fps):
            if value is not None and not value > 0:
                raise ValueError("Capture dimensions and FPS must be positive")
    except ValueError as exc:
        parser.error(str(exc))
    from PyQt5.QtWidgets import QApplication

    from frontend.ui.main_screen import MainWindow

    app = QApplication([sys.argv[0]])
    window = MainWindow(
        video_source=parse_source(args.source),
        model_path=args.model,
        actor_path=args.actor,
        serial_port=args.serial_port,
        runtime=runtime,
        capture_backend=args.capture_backend,
        source_mode=args.source_mode,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        capture_fps=args.capture_fps,
        invert_pan=args.invert_pan,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
