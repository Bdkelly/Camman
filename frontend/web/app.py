"""Serve Camman on a phone, tablet or local touchscreen browser."""

import argparse
import os
import secrets
from pathlib import Path

from backend.capture import add_capture_arguments
from backend.runtime import add_runtime_arguments, resolve_runtime
from backend.web.service import CameraConfig, CameraService


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 for LAN access")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", help="Detector checkpoint; omitted enables preview/manual only")
    parser.add_argument("--actor", help="Versioned actor; otherwise proportional control")
    parser.add_argument("--serial-port", help="Explicit serial port or auto; omitted means dry-run")
    parser.add_argument("--invert-pan", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.98)
    parser.add_argument("--inference-fps", type=float)
    parser.add_argument("--preview-fps", type=float, default=8)
    parser.add_argument("--preview-width", type=int, default=960)
    add_capture_arguments(parser)
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    try:
        import uvicorn

        from backend.web.api import create_app
    except ImportError:
        parser.error('Install the web extra: python -m pip install -e ".[web]"')
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be 1..65535")
        runtime = resolve_runtime(
            profile=args.profile,
            device=args.device,
            precision=args.precision,
            detector_size=args.detector_size,
            inference_fps=args.inference_fps,
            cpu_threads=args.cpu_threads,
            preview_fps=args.preview_fps,
        )
        config = CameraConfig(
            **{name: getattr(args, name) for name in CameraConfig.__dataclass_fields__}
        )
        token = os.environ.get("CAMMAN_WEB_TOKEN") or secrets.token_urlsafe(24)
        app = create_app(CameraService(config, runtime), token, Path(__file__).parent / "static")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Camman web: http://{args.host}:{args.port} (use the Jetson's LAN IP on your phone)")
    if "CAMMAN_WEB_TOKEN" not in os.environ:
        print(f"Pairing key: {token}", flush=True)
    else:
        print("Pair using your configured CAMMAN_WEB_TOKEN.", flush=True)
    print("Use a trusted private network; HTTP is not encrypted. No internet is required.")
    # One worker owns the camera/serial. Never enable reload or multiple workers.
    uvicorn.run(
        app, host=args.host, port=args.port, workers=1, access_log=False, proxy_headers=False
    )


if __name__ == "__main__":
    main()
