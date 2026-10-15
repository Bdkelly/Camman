"""Single-owner camera service. Control deadlines never wait for detector inference."""

import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2

from backend.capture import FrameReader, is_live_source, open_capture, parse_source
from backend.control import ControlSpec
from backend.detection import FrameTransform, get_ball_detection
from backend.hardware.serial_connection import open_connection, verify_velocity_controller
from backend.models import load_model_from_path
from backend.policy import ActorPolicy
from backend.tracking import TrackingController


class ControlError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class CameraConfig:
    source: str = "0"
    source_mode: str = "auto"
    capture_backend: str = "auto"
    capture_width: int | None = None
    capture_height: int | None = None
    capture_fps: float | None = None
    model: str | None = None
    actor: str | None = None
    serial_port: str | None = None
    invert_pan: bool = False
    confidence: float = 0.98
    preview_fps: float = 8
    preview_width: int = 960

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be 0..1")
        if not 1 <= self.preview_fps <= 15 or not 160 <= self.preview_width <= 1920:
            raise ValueError("Preview FPS must be 1..15 and width 160..1920")
        for value in (self.capture_width, self.capture_height, self.capture_fps):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Capture dimensions and FPS must be positive")
        if self.actor and not self.model:
            raise ValueError("An actor requires a detector --model")


class CameraService:
    lease_seconds = 2.0

    def __init__(self, config, runtime, *, clock=time.monotonic):
        self.config, self.runtime, self.clock = config, runtime, clock
        self.lock = threading.RLock()
        self.halt = threading.Event()
        self.controller = TrackingController(invert_pan=config.invert_pan, clock=clock)
        self.ser = self.policy = self.detector = None
        self.ready = self.tracking = False
        self.phase = "starting"
        self.error = None
        self.logs = deque(maxlen=100)
        self.lease = None
        self.deadline = 0
        self.generation = 0
        self.jpeg = None
        self.preview_at = None
        self.inference_ms = None
        self.target = False
        self.threads = []

    def log(self, message):
        with self.lock:
            self.logs.append(str(message))

    def _stop(self, reason):
        self.tracking = False
        self.target = False
        self.generation += 1
        self.controller.stop(self.ser, force=True, reset=True, log=self.log)
        self.log(reason)

    def _expire(self):
        if self.lease is not None and self.clock() >= self.deadline:
            self.lease = None
            self._stop("Control connection expired; motion stopped")

    def acquire(self):
        with self.lock:
            self._expire()
            if self.lease is not None:
                raise ControlError(
                    "Another browser has control. Release it or wait for its timeout."
                )
            self.lease = secrets.token_urlsafe(32)
            self.deadline = self.clock() + self.lease_seconds
            return {"lease": self.lease, "expires_in": self.lease_seconds}

    def _require(self, lease):
        self._expire()
        if not lease or self.lease is None or not secrets.compare_digest(lease, self.lease):
            raise ControlError("Control expired or belongs to another browser", 403)

    def heartbeat(self, lease):
        with self.lock:
            self._require(lease)
            self.deadline = self.clock() + self.lease_seconds

    def release(self, lease):
        with self.lock:
            self._require(lease)
            self.lease = None
            self._stop("Control released")

    def command(self, action, lease=None):
        with self.lock:
            # Any paired observer may stop, invalidating in-flight motion requests.
            if action == "stop":
                self.lease = None
                self._stop("Stop requested; take control again to resume")
                return
            self._require(lease)
            if not self.ready or self.halt.is_set():
                raise ControlError("Camera service is not ready")
            if action == "start":
                if self.detector is None:
                    raise ControlError("Restart the server with --model to enable tracking")
                if not self.tracking:
                    self._stop("Starting automatic tracking")
                    self.tracking = True
            elif action in {"left", "right"}:
                if self.tracking:
                    raise ControlError("Stop tracking before manual positioning")
                self.controller.manual(action.title(), self.ser, log=self.log, duration=0.2)
            elif action == "manual-stop" and not self.tracking:
                self.controller.manual("Stop", self.ser, log=self.log)
            elif action != "manual-stop":
                raise ControlError("Unknown command", 422)

    def status(self):
        with self.lock:
            return {
                "phase": self.phase,
                "ready": self.ready,
                "tracking": self.tracking,
                "dry_run": self.ser is None,
                "detector_loaded": self.detector is not None,
                "policy": "actor" if self.policy else "proportional",
                "control_held": self.lease is not None and self.clock() < self.deadline,
                "preview_age_s": None
                if self.preview_at is None
                else self.clock() - self.preview_at,
                "inference_ms": self.inference_ms,
                "target_detected": self.target,
                "runtime": self.runtime.describe(),
                "error": self.error,
                "logs": list(self.logs),
            }

    def preview(self):
        with self.lock:
            return self.jpeg, (None if self.preview_at is None else self.clock() - self.preview_at)

    def start(self):
        self.threads = [
            threading.Thread(target=self._control_loop, name="web-control", daemon=True),
            threading.Thread(target=self._camera_loop, name="web-camera", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _fail(self, exc):
        with self.lock:
            self.ready = False
            self.phase = "error"
            self.error = str(exc)
            self.lease = None
            self.halt.set()
            try:
                self._stop(f"Service error: {exc}")
            except Exception as stop_error:
                self.log(f"Stop write failed: {stop_error}; controller watchdog must stop motion")

    def tick(self):
        with self.lock:
            self._expire()
            self.controller.tick(self.ser, enabled=self.tracking, log=self.log)
            if (
                self.controller.observations.last_seen is None
                or self.clock() - self.controller.observations.last_seen
                >= self.controller.spec.lost_target_timeout_s
            ):
                self.target = False

    def _control_loop(self):
        try:
            while not self.halt.wait(0.02):
                self.tick()
        except Exception as exc:
            self._fail(exc)

    def apply_detection(self, boxes, packet, generation):
        with self.lock:
            self._expire()
            if not self.tracking or generation != self.generation or self.halt.is_set():
                return
            h, w = packet.image.shape[:2]
            self.controller.update(
                boxes,
                self.ser,
                w,
                h,
                agent=self.policy,
                log=self.log,
                observed_at=packet.captured_at,
            )
            self.target = (
                bool(boxes)
                and self.clock() - packet.captured_at <= self.controller.spec.max_frame_age_s
            )

    def _camera_loop(self):
        cap = reader = serial = None
        try:
            self.runtime.apply()
            cfg = self.config
            policy = ActorPolicy(cfg.actor, "cpu") if cfg.actor else None
            detector = (
                load_model_from_path(
                    cfg.model, self.runtime.device, detector_size=self.runtime.detector_size
                )
                if cfg.model
                else None
            )
            if self.halt.is_set():
                return
            source = parse_source(cfg.source)
            live = is_live_source(source, cfg.source_mode, cfg.capture_backend)
            cap = open_capture(
                source,
                backend=cfg.capture_backend,
                live=live,
                width=cfg.capture_width,
                height=cfg.capture_height,
                fps=cfg.capture_fps,
            )
            fps = cap.get(cv2.CAP_PROP_FPS)
            period = 0 if live else 1 / (fps if math.isfinite(fps) and 0 < fps < 240 else 30)
            serial = open_connection(cfg.serial_port)
            if cfg.serial_port and serial is None:
                raise RuntimeError("Requested serial controller was not found")
            verify_velocity_controller(serial)
            with self.lock:
                if self.halt.is_set():
                    return
                self.policy, self.detector, self.ser = policy, detector, serial
                self.controller = TrackingController(
                    policy.spec if policy else ControlSpec(),
                    invert_pan=cfg.invert_pan,
                    clock=self.clock,
                )
                self.controller.stop(serial, force=True, reset=True)
                self.ready, self.phase = True, "ready"
                self.log("Preview ready. Take control to start tracking.")
            reader = FrameReader(cap, live=live).start()
            transform = FrameTransform()
            next_frame = next_detect = next_preview = 0
            while not self.halt.is_set():
                now = self.clock()
                if now < next_frame:
                    self.halt.wait(min(0.01, next_frame - now))
                    continue
                packet = reader.read(timeout=0.05)
                if packet is None:
                    if reader.ended:
                        if reader.error:
                            raise RuntimeError(reader.error)
                        with self.lock:
                            self.ready, self.phase, self.lease = False, "ended", None
                            self._stop("Video ended; restart the server to replay")
                        break
                    continue
                next_frame = now + period
                with self.lock:
                    tracking, generation = self.tracking, self.generation
                frame = packet.image
                if tracking and now >= next_detect:
                    started = self.clock()
                    boxes, frame = get_ball_detection(
                        detector,
                        frame,
                        transform,
                        self.runtime.device,
                        confidence_threshold=cfg.confidence,
                        precision=self.runtime.precision,
                    )
                    self.apply_detection(boxes, packet, generation)
                    with self.lock:
                        self.inference_ms = (self.clock() - started) * 1000
                    next_detect = started + (
                        1 / self.runtime.inference_fps if self.runtime.inference_fps else 0
                    )
                if self.clock() >= next_preview:
                    h, w = frame.shape[:2]
                    if w > cfg.preview_width:
                        frame = cv2.resize(
                            frame, (cfg.preview_width, max(1, round(h * cfg.preview_width / w)))
                        )
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        with self.lock:
                            self.jpeg, self.preview_at = jpg.tobytes(), packet.captured_at
                    next_preview = self.clock() + 1 / cfg.preview_fps
        except Exception as exc:
            self._fail(exc)
        finally:
            with self.lock:
                try:
                    self.controller.stop(serial, force=serial is not None, reset=True)
                except Exception as exc:
                    self.log(f"Final stop failed: {exc}")
                self.ser = None
                if serial is not None:
                    serial.close()
            if reader is not None:
                reader.close()
            elif cap is not None:
                cap.release()

    def close(self):
        self.halt.set()
        with self.lock:
            self.ready = False
            self.lease = None
            try:
                self._stop("Server shutting down")
            except Exception as exc:
                self.log(f"Shutdown stop failed: {exc}")
        for thread in self.threads:
            thread.join(timeout=3)
