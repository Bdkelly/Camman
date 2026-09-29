"""Bounded capture without Qt. Live inputs drop old frames; files keep every frame."""

import re
import threading
import time
from dataclasses import dataclass

import cv2

CAPTURE_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "gstreamer": cv2.CAP_GSTREAMER,
    "v4l2": cv2.CAP_V4L2,
    "ffmpeg": cv2.CAP_FFMPEG,
}


def has_gstreamer():
    return bool(re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()))


def is_live_source(source, mode="auto", backend="auto"):
    if mode not in {"auto", "live", "file"}:
        raise ValueError("Source mode must be auto, live or file")
    if mode != "auto":
        return mode == "live"
    return (
        isinstance(source, int)
        or str(source).startswith(
            ("/dev/video", "rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")
        )
        or (backend == "gstreamer" and "!" in str(source))
    )


def open_capture(source, *, backend="auto", live=False, width=None, height=None, fps=None):
    api = CAPTURE_BACKENDS[backend]
    if backend == "gstreamer" and not has_gstreamer():
        raise RuntimeError(
            "OpenCV has no GStreamer support. Use Jetson's system OpenCV; "
            "see backend/JETSON.md. The pip headless wheel does not provide this backend."
        )
    if backend in {"ffmpeg", "gstreamer"}:
        cap = cv2.VideoCapture(
            source,
            api,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                2000,
            ],
        )
    elif backend == "auto":
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, api)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video source with {backend}. Check the source, "
                "camera permissions, pipeline and installed OpenCV backends."
            )
        if live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; not all drivers support it.
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, width),
            (cv2.CAP_PROP_FRAME_HEIGHT, height),
            (cv2.CAP_PROP_FPS, fps),
        ):
            if value is not None:
                cap.set(prop, value)
        return cap
    except Exception:
        cap.release()
        raise


@dataclass(frozen=True)
class CapturedFrame:
    image: object
    captured_at: float


class FrameReader:
    """Own read/release on one thread, with at most one waiting frame.

    Shutdown never releases a VideoCapture while another thread is reading it.
    FFmpeg/GStreamer inputs use read timeouts. A stuck native camera driver can
    outlive the join timeout; its daemon reader releases capture when read returns.
    """

    def __init__(self, cap, *, live):
        self.cap = cap
        self.live = live
        self.dropped = 0
        self.error = None
        self._slot = None
        self._ended = False
        self._stop = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="camman-capture", daemon=True)

    def start(self):
        self._thread.start()
        return self

    @property
    def ended(self):
        with self._condition:
            return self._ended and self._slot is None

    def read(self, timeout=0.1):
        with self._condition:
            self._condition.wait_for(
                lambda: self._slot is not None or self._ended or self._stop, timeout
            )
            frame, self._slot = self._slot, None
            self._condition.notify_all()
            return frame

    def close(self, timeout=2.5):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self):
        try:
            while True:
                with self._condition:
                    # Files apply back pressure rather than racing to EOF.
                    self._condition.wait_for(lambda: self._stop or self.live or self._slot is None)
                    if self._stop:
                        break
                ok, image = self.cap.read()
                if not ok:
                    break
                frame = CapturedFrame(image, time.monotonic())
                with self._condition:
                    if self._stop:
                        break
                    if self._slot is not None:
                        self.dropped += 1
                    self._slot = frame
                    self._condition.notify_all()
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                self.cap.release()
            except Exception as exc:
                self.error = self.error or f"Capture release failed: {exc}"
            with self._condition:
                self._ended = True
                self._condition.notify_all()


def add_capture_arguments(parser, *, playback=True):
    parser.add_argument("--source", default="0", help="Camera index, video path, URL or pipeline")
    parser.add_argument("--capture-backend", choices=tuple(CAPTURE_BACKENDS), default="auto")
    if playback:
        parser.add_argument(
            "--source-mode",
            choices=("auto", "live", "file"),
            default="auto",
            help="Live keeps newest frames; file preserves all frames. Pipelines default live",
        )
    parser.add_argument(
        "--capture-width", type=int, help="Requested camera width; driver may ignore"
    )
    parser.add_argument(
        "--capture-height", type=int, help="Requested camera height; driver may ignore"
    )
    parser.add_argument("--capture-fps", type=float, help="Requested camera FPS; driver may ignore")


def parse_source(value):
    return int(value) if value.isdecimal() else value
