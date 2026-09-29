"""Worker bridging UI signals to the backend, owning camera/serial resources."""

import cv2
from PyQt5.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage

from backend.capture import FrameReader, is_live_source, open_capture
from backend.config import models_directory
from backend.detection import FrameTransform
from backend.hardware.serial_connection import open_connection, send_agent_command
from backend.models import load_model_from_path
from backend.policy import ActorPolicy
from backend.runtime import resolve_runtime
from backend.tracking import TrackingController
from frontend.video import videorun


class VideoThread(QThread):
    command_log_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        *,
        video_source=0,
        model_path=None,
        actor_path=None,
        serial_port=None,
        device=None,
        runtime=None,
        capture_backend="auto",
        source_mode="auto",
        capture_width=None,
        capture_height=None,
        capture_fps=None,
    ):
        super().__init__(parent)
        self._run_flag = True
        self.inference_active = False
        self.agent_active = False
        self.mutex = QMutex()
        self.agent = None
        self.ser = None
        self.command_interval = 1.0
        self.model = None
        self.video_source = video_source
        self.actor_path = actor_path
        self.serial_port = serial_port
        self.runtime = runtime or resolve_runtime(device=device)
        self.device = self.runtime.device
        self.capture_backend = capture_backend
        self.live = is_live_source(video_source, source_mode, capture_backend)
        self.capture_options = dict(width=capture_width, height=capture_height, fps=capture_fps)
        self._pending_model_path = model_path
        self._pending_command = None
        self._preview = None
        self.controller = TrackingController()

    def run(self):
        cap = None
        reader = None
        try:
            self.runtime.apply()
            self.command_log_signal.emit(self.runtime.describe())
            if self.runtime.profile == "jetson" and self.device.type != "cuda":
                self.command_log_signal.emit(
                    "Jetson is using CPU. Install a matching CUDA-enabled PyTorch/torchvision build."
                )
            cap = open_capture(
                self.video_source,
                backend=self.capture_backend,
                live=self.live,
                **self.capture_options,
            )
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.command_log_signal.emit(
                f"Capture: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):g}x"
                f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):g}, {fps:g} FPS; "
                f"{'live (newest frame)' if self.live else 'file (all frames)'}"
            )
            with QMutexLocker(self.mutex):
                if self._pending_model_path is None:
                    self._pending_model_path = next(
                        iter(sorted(models_directory().glob("*.pth"))), None
                    )
            self.load_pending_model()
            if self.model is None:
                self.command_log_signal.emit(
                    "Video preview ready. Load a detector through Models to enable inference."
                )
            if self.actor_path:
                try:
                    # Keep the tiny policy on CPU on Jetson, avoiding another
                    # GPU round trip before its result is sent over serial.
                    actor_device = "cpu" if self.runtime.profile == "jetson" else self.device
                    self.agent = ActorPolicy(self.actor_path, actor_device)
                except Exception as exc:
                    self.command_log_signal.emit(f"Actor unavailable: {exc}. Using basic tracking.")
            try:
                self.ser = open_connection(self.serial_port)
            except Exception as exc:
                self.command_log_signal.emit(f"Serial connection unavailable: {exc}")
            reader = FrameReader(cap, live=self.live).start()
            videorun(self, reader, FrameTransform(), fps)
        except Exception as exc:
            self.command_log_signal.emit(f"Video stopped: {exc}")
        finally:
            self._run_flag = False
            if reader is not None:
                if not reader.close():
                    self.command_log_signal.emit(
                        "Capture driver is still returning from read; release is deferred."
                    )
            elif cap is not None:
                cap.release()
            if self.ser is not None:
                self.ser.close()
                self.ser = None

    def stop(self):
        self._run_flag = False
        self.wait()

    @pyqtSlot(str)
    def update_model(self, model_path):
        # The worker loads weights between frames, keeping disk/GPU work off the UI thread.
        with QMutexLocker(self.mutex):
            self._pending_model_path = model_path
        self.command_log_signal.emit(f"Requesting model update: {model_path}")

    def load_pending_model(self):
        with QMutexLocker(self.mutex):
            path = self._pending_model_path
            self._pending_model_path = None
        if path is None:
            return
        try:
            model = load_model_from_path(
                path, self.device, detector_size=self.runtime.detector_size
            )
            with QMutexLocker(self.mutex):
                self.model = model
            self.command_log_signal.emit(f"Model loaded from {path}")
        except Exception as exc:
            self.command_log_signal.emit(f"Failed to load model: {exc}")

    @pyqtSlot(bool)
    def toggle_inference(self, state):
        with QMutexLocker(self.mutex):
            self.inference_active = state
        self.command_log_signal.emit(f"--- Inference {'STARTED' if state else 'STOPPED'} ---")

    @pyqtSlot(bool)
    def toggle_agent(self, state):
        with QMutexLocker(self.mutex):
            self.agent_active = state
        self.command_log_signal.emit(f"--- CamMan Agent {'STARTED' if state else 'STOPPED'} ---")

    @pyqtSlot(float)
    def set_command_interval(self, interval):
        with QMutexLocker(self.mutex):
            self.command_interval = max(0.0, interval)
        self.command_log_signal.emit(f"Command Interval set to: {self.command_interval:.2f}s")

    @pyqtSlot(str)
    def request_manual_command(self, command):
        if command not in {"Left", "Right", "Stop"}:
            raise ValueError("Unknown manual command")
        # Coalesce clicks rather than queuing stale motor commands behind inference.
        with QMutexLocker(self.mutex):
            self._pending_command = command

    def send_pending_command(self):
        with QMutexLocker(self.mutex):
            command, self._pending_command = self._pending_command, None
        if command is not None:
            try:
                send_agent_command(self.ser, command, self.command_log_signal.emit)
            except Exception as exc:
                self.command_log_signal.emit(f"Manual command failed: {exc}")

    def publish_preview(self, frame):
        image = self._convert_cv_qt(frame)
        with QMutexLocker(self.mutex):
            self._preview = image

    def take_preview(self):
        # The UI polls one slot: a slow window never queues full QImages.
        with QMutexLocker(self.mutex):
            image, self._preview = self._preview, None
        return image

    def _convert_cv_qt(self, cv_img):
        height, width = cv_img.shape[:2]
        scale = min(640 / width, 480 / height)
        small = cv2.resize(
            cv_img,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, width * channels, QImage.Format_RGB888)
        return image.copy()
