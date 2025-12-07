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
AGENT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'actormodels', 'agent_model.pth'))

class VideoThread(QThread):
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
   
        success, model, transform, ser_connection = init_video_comp(self)
        
        if not success:
            self._run_flag = False
            self.command_log_signal.emit("Initialization failed.")
            return
        self.mutex.lock()
        self.model = model
        self.ser = ser_connection

        self.mutex.unlock()  

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
        self._run_flag = False
        self.wait()

    def update_model(self, model_path):
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
        self.mutex.lock()
        self.inference_active = state
        self.mutex.unlock()
        self.command_log_signal.emit(f"--- Inference {'STARTED' if state else 'STOPPED'} ---")

    def toggle_agent(self, state):
        self.mutex.lock()
        self.agent_active = state
        self.mutex.unlock()
        self.command_log_signal.emit(f"--- CamMan Agent {'STARTED' if state else 'STOPPED'} ---")


    
    @pyqtSlot(float)
    def set_command_interval(self, interval):
        self.mutex.lock()
        self.command_interval = interval
        self.mutex.unlock()
        self.command_log_signal.emit(f"Command Interval set to: {interval:.2f}s")
        
    def _convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return p