"""Desktop entry point. Parse arguments before importing optional Qt dependencies."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Camman ball tracking application")
    parser.add_argument("--source", default="0", help="Camera index or video file (default: 0)")
    parser.add_argument("--model", help="Detector .pth checkpoint")
    parser.add_argument("--actor", help="Optional camera-control actor .pth checkpoint")
    parser.add_argument("--serial-port", help="ESP32 port, e.g. COM3, /dev/ttyUSB0, or auto")
    parser.add_argument(
        "--device", default=None, help="Torch device; default: CUDA if available, otherwise CPU"
    )
    args = parser.parse_args(argv)
    from PyQt5.QtWidgets import QApplication

    from frontend.ui.main_screen import MainWindow

    source = int(args.source) if args.source.isdecimal() else args.source
    app = QApplication([sys.argv[0]])
    window = MainWindow(
        video_source=source,
        model_path=args.model,
        actor_path=args.actor,
        serial_port=args.serial_port,
        device=args.device,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
