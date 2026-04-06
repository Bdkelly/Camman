"""
video_threads — Background QThread for video capture, detection, and agent control.

``VideoThread`` runs the entire perception-action loop in a dedicated thread so
that the PyQt5 GUI main thread remains responsive.  On each iteration it:

1. Reads a frame from a video file or camera (``cv2.VideoCapture``).
2. Optionally runs Faster R-CNN ball detection (toggled by ``inference_active``).
3. Optionally queries the DDPG agent for a pan command (toggled by ``agent_active``).
4. Converts the annotated frame to a ``QImage`` and emits it to the GUI.

Thread safety is maintained via a ``QMutex`` that guards shared state
(``model``, ``inference_active``, ``agent_active``, ``command_interval``)
that can be modified from the main thread via Qt signals.
"""

import time
import os
import cv2
import torch
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex, Qt
from PyQt5.QtGui import QImage

from RLAgent.RLAgent import RLAgent
from RLAgent.camController import CameraControlEnv

try:
    from guiapp.utils.vidpro import videorun, init_video_comp, load_model_from_path
    from guiapp.utils.models import get_fasterrcnn_model_single_class as fmodel
except ImportError:
    from utils.vidpro import videorun, init_video_comp, load_model_from_path
    from utils.models import get_fasterrcnn_model_single_class as fmodel

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']
AGENT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'agentModel', 'agentModel.pth'))

USE_RECORDED_VIDEO = True
RECORDED_VIDEO_PATH = r"/Users/Ben/Documents/dever/python/ptorch/data/SmallVideo.MP4"

class VideoThread(QThread):
    """QThread that drives the perception-action loop for the Camman GUI.

    Emits two signals:

    - ``change_pixmap_signal(QImage)``: Delivers each processed frame to the
      ``MainWindow`` image label for display.
    - ``command_log_signal(str)``: Delivers log messages (serial commands, status
      updates, errors) to the ``MainWindow`` command log widget.

    The thread is started by ``MainWindow.__init__`` and stopped on window close
    via ``stop()``.

    Attributes:
        _run_flag (bool): Loop sentinel; set to False by ``stop()`` to exit gracefully.
        inference_active (bool): Whether ball detection is currently running.
        agent_active (bool): Whether the DDPG agent is currently issuing commands.
        mutex (QMutex): Guards all shared state modified from the main thread.
        agent (RLAgent | None): Loaded DDPG agent; None if weights not found.
        ser (serial.Serial | None): Serial connection to the ESP32; None if not connected.
        command_interval (float): Minimum seconds between serial commands (default 1.0).
        last_command_time (float): Timestamp of the last command sent.
        prev_action (float): Most recent pan action for state construction.
        model (torch.nn.Module | None): Loaded Faster R-CNN detection model.
        device (torch.device): Compute device (CUDA if available, else CPU).
    """

    change_pixmap_signal = pyqtSignal(QImage)
    command_log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True
        self.inference_active = False
        self.agent_active = False
        self.mutex = QMutex()
        self.agent = None

        self.ser = None
        self.command_interval = 1.0
        self.last_command_time = time.time()
        self.prev_action = 0.0

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        """Thread entry point: initialise components and start the video loop.

        Calls ``init_video_comp`` to load the detection model, build the
        image transform, and attempt a serial connection. Then opens the
        video source and the agent model before delegating to ``videorun``.
        Emits a log message and exits early if any step fails.
        """
        success, model, transform, ser_connection = init_video_comp(self)
        
        if not success:
            self._run_flag = False
            self.command_log_signal.emit("Initialization failed.")
            return
        self.mutex.lock()
        self.model = model
        self.ser = ser_connection

        self.mutex.unlock()  
        if USE_RECORDED_VIDEO:
            cap = cv2.VideoCapture(RECORDED_VIDEO_PATH)
            self.command_log_signal.emit(f"Playing video: {RECORDED_VIDEO_PATH}")
        else:
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.command_log_signal.emit("Error: Could not open video stream.")
            self._run_flag = False
            return
            
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_size = 4  
        action_size = 1
        max_action = 1.0  
        self.agent = RLAgent(state_size, action_size, max_action, self.device)
        try:
            if os.path.exists(AGENT_MODEL_PATH):
                self.agent.actor_local.load_state_dict(torch.load(AGENT_MODEL_PATH, map_location=self.device))
                self.command_log_signal.emit(f"Agent model loaded from {AGENT_MODEL_PATH}")
            else:
                 self.command_log_signal.emit(f"Agent model not found at {AGENT_MODEL_PATH}. Agent disabled.")
                 self.agent = None
        except Exception as e:
            self.command_log_signal.emit(f"Failed to load agent model: {e}")
            self._run_flag = False
            return

        videorun(self, cap, W, H, self.model, transform, self.device, self.ser)

    def stop(self):
        """Signal the video loop to exit and block until the thread finishes."""
        self._run_flag = False
        self.wait()

    def update_model(self, model_path):
        """Hot-swap the detection model while the thread is running.

        Loads the new model from ``model_path`` on the calling (main) thread
        and swaps it under the mutex so the video loop picks it up safely.

        Args:
            model_path (str): Filesystem path to the new ``.pth`` weights file.
        """
        self.command_log_signal.emit(f"Requesting model update: {model_path}")
        try:
            new_model = load_model_from_path(model_path, self.device, self)
            if new_model:
                self.mutex.lock()
                self.model = new_model
                self.mutex.unlock()
                self.command_log_signal.emit("Model updated successfully.")
            else:
                self.command_log_signal.emit("Failed to load new model.")
        except Exception as e:
            self.command_log_signal.emit(f"Exception during model update: {e}")

    @pyqtSlot(bool)
    def toggle_inference(self, state):
        """Enable or disable ball detection under the mutex.

        Args:
            state (bool): True to start inference, False to stop.
        """
        self.mutex.lock()
        self.inference_active = state
        self.mutex.unlock()
        self.command_log_signal.emit(f"--- Inference {'STARTED' if state else 'STOPPED'} ---")

    def toggle_agent(self, state):
        """Enable or disable the DDPG agent under the mutex.

        Args:
            state (bool): True to start the agent, False to stop.
        """
        self.mutex.lock()
        self.agent_active = state
        self.mutex.unlock()
        self.command_log_signal.emit(f"--- CamMan Agent {'STARTED' if state else 'STOPPED'} ---")

    @pyqtSlot(float)
    def set_command_interval(self, interval):
        """Update the minimum time between serial commands under the mutex.

        Args:
            interval (float): New interval in seconds.
        """
        self.mutex.lock()
        self.command_interval = interval
        self.mutex.unlock()
        self.command_log_signal.emit(f"Command Interval set to: {interval:.2f}s")

    def _convert_cv_qt(self, cv_img):
        """Convert a BGR OpenCV image to a scaled QImage for GUI display.

        Args:
            cv_img (np.ndarray): BGR image array of shape (H, W, 3).

        Returns:
            QImage: RGB image scaled to 640×480 with aspect ratio preserved.
        """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return p