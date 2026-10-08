"""Run ball detection and camera control without a GUI; no port means dry-run."""

import argparse
import math
import time

import cv2

from backend.capture import (
    FrameReader,
    add_capture_arguments,
    is_live_source,
    open_capture,
    parse_source,
)
from backend.control import ControlSpec
from backend.detection import FrameTransform, get_ball_detection
from backend.hardware.serial_connection import open_connection, verify_velocity_controller
from backend.models import load_model_from_path
from backend.policy import ActorPolicy
from backend.runtime import add_runtime_arguments, resolve_runtime
from backend.tracking import TrackingController


def run_tracking(
    reader,
    detector,
    runtime,
    controller,
    *,
    policy=None,
    ser=None,
    fps=30,
    confidence=0.98,
    max_frames=None,
    log=print,
):
    """The caller owns capture/serial; this loop always stops motion on exit."""
    period = 0.0 if reader.live else 1 / (fps if math.isfinite(fps) and 0 < fps < 240 else 30)
    if runtime.inference_fps:
        period = max(period, 1 / runtime.inference_fps)
    next_frame = 0.0
    transform = FrameTransform()
    count = 0
    try:
        while max_frames is None or count < max_frames:
            controller.tick(ser, log=log)
            now = time.monotonic()
            if now < next_frame:
                time.sleep(min(0.01, next_frame - now))
                continue
            packet = reader.read(timeout=0.05)
            if packet is None:
                if reader.ended:
                    if reader.error:
                        raise RuntimeError(f"Capture failed: {reader.error}")
                    break
                continue
            started = time.monotonic()
            boxes, _ = get_ball_detection(
                detector,
                packet.image,
                transform,
                runtime.device,
                confidence_threshold=confidence,
                draw=False,
                precision=runtime.precision,
            )
            height, width = packet.image.shape[:2]
            controller.update(
                boxes, ser, width, height, agent=policy, log=log, observed_at=packet.captured_at
            )
            count += 1
            next_frame = started + period
    finally:
        controller.stop(ser, log=log, force=ser is not None, reset=True)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ball detector checkpoint")
    parser.add_argument("--actor", help="Versioned Camman actor; otherwise proportional control")
    parser.add_argument("--serial-port", help="Explicit port or auto; omitted means dry-run")
    parser.add_argument(
        "--invert-pan", action="store_true", help="Reverse the wired motor direction"
    )
    parser.add_argument("--confidence", type=float, default=0.98)
    parser.add_argument("--frames", type=int, help="Stop after this many processed frames")
    parser.add_argument("--inference-fps", type=float, help="Detector FPS ceiling; 0 is unlimited")
    add_capture_arguments(parser)
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    cap = reader = ser = controller = None
    try:
        if not 0 <= args.confidence <= 1 or (args.frames is not None and args.frames < 1):
            raise ValueError("Confidence must be 0..1 and frames must be positive")
        for value in (args.capture_width, args.capture_height, args.capture_fps):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Capture dimensions and FPS must be positive")
        runtime = resolve_runtime(
            profile=args.profile,
            device=args.device,
            precision=args.precision,
            detector_size=args.detector_size,
            cpu_threads=args.cpu_threads,
            inference_fps=args.inference_fps,
        )
        runtime.apply()
        print(runtime.describe())
        # The actor is tiny; CPU avoids synchronizing another GPU operation.
        policy = ActorPolicy(args.actor, "cpu") if args.actor else None
        controller = TrackingController(
            policy.spec if policy else ControlSpec(), invert_pan=args.invert_pan
        )
        detector = load_model_from_path(
            args.model, runtime.device, detector_size=runtime.detector_size
        )
        source = parse_source(args.source)
        live = is_live_source(source, args.source_mode, args.capture_backend)
        cap = open_capture(
            source,
            backend=args.capture_backend,
            live=live,
            width=args.capture_width,
            height=args.capture_height,
            fps=args.capture_fps,
        )
        fps = cap.get(cv2.CAP_PROP_FPS)
        ser = open_connection(args.serial_port)
        if args.serial_port and ser is None:
            raise RuntimeError("No controller found for the requested serial port")
        verify_velocity_controller(ser)
        print(
            f"{'Controller connected' if ser else 'Dry-run'}; "
            f"pan limit {controller.spec.max_pan_speed_deg_s:g} degrees/s; "
            f"FOV {controller.spec.camera_hfov_deg:g} degrees"
        )
        reader = FrameReader(cap, live=live).start()
        count = run_tracking(
            reader,
            detector,
            runtime,
            controller,
            policy=policy,
            ser=ser,
            fps=fps,
            confidence=args.confidence,
            max_frames=args.frames,
        )
        print(f"Processed {count} frames; dropped {reader.dropped} capture frames.")
    except KeyboardInterrupt:
        print("Tracking interrupted.")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Tracking failed: {exc}\n")
    finally:
        try:
            if controller is not None and ser is not None:
                controller.stop(ser, force=True)
        finally:
            if ser is not None:
                ser.close()
            if reader is not None:
                if not reader.close():
                    print("Capture driver is still returning from read; release is deferred.")
            elif cap is not None:
                cap.release()


if __name__ == "__main__":
    main()
