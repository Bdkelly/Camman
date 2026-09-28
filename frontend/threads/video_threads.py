"""Worker bridging UI signals to the backend, owning camera/serial resources."""

import cv2
import torch
from PyQt5.QtCore import QMutex, QMutexLocker, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage

from backend.config import models_directory
from backend.detection import FrameTransform
from backend.hardware.serial_connection import open_connection
from backend.models import load_model_from_path
from backend.policy import ActorPolicy
from backend.tracking import TrackingController
from frontend.video import videorun


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    command_log_signal = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        *,
        video_source=0,
        model_path=None,
        actor_path=None,
        serial_port=None,
        device=None,
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
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._pending_model_path = model_path
        self.controller = TrackingController()

    def run(self):
        cap = None
        try:
            self.command_log_signal.emit(f"Using device: {self.device}")
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.video_source}")
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
                    self.agent = ActorPolicy(self.actor_path, self.device)
                except Exception as exc:
                    self.command_log_signal.emit(f"Actor unavailable: {exc}. Using basic tracking.")
            try:
                self.ser = open_connection(self.serial_port)
            except Exception as exc:
                self.command_log_signal.emit(f"Serial connection unavailable: {exc}")
            videorun(self, cap, FrameTransform())
        except Exception as exc:
            self.command_log_signal.emit(f"Video stopped: {exc}")
        finally:
            self._run_flag = False
            if cap is not None:
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
            model = load_model_from_path(path, self.device)
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

    def _convert_cv_qt(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, width * channels, QImage.Format_RGB888)
        return image.scaled(640, 480, Qt.KeepAspectRatio).copy()
