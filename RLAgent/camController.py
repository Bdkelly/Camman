#Agent Calls
from .ballfind import get_ball_detection
from .reward import RewardSystem
#Standard Calls
import cv2
import numpy as np
import torch
import serial


class CameraControlEnv:
    def __init__(self, cap, detection_model, transform, device, 
                 frame_center_x, frame_center_y, max_action, 
                 reward_system: RewardSystem):
        
        self.cap = cap
        self.detection_model = detection_model
        self.transform = transform
        self.device = device
        
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FRAME_CENTER = (self.W // 2, self.H // 2)
        
        self.max_action = max_action 
        self.reward_system = reward_system

        self.ser = None 
        
        self.current_frame = None

        self.dyn_center_x = self.FRAME_CENTER[0] 
        self.dyn_center_y = self.FRAME_CENTER[1]
        
        self.window_width = int(0.25 * self.W)
        self.window_height = self.H
        
        self.max_center_shift_x = self.W * 0.25 
        self.max_center_shift_y = self.H * 0.25 
    
    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, self.current_frame = self.cap.read()
        self.reward_system.reset()
        
        self.dyn_center_x = self.FRAME_CENTER[0]
        self.dyn_center_y = self.FRAME_CENTER[1]

        ball_x, ball_y, frame_with_detections, is_detected = self.detect_ball()
        dx = (ball_x - self.dyn_center_x) / self.W 
        dy = (ball_y - self.dyn_center_y) / self.H 
        
        initial_state = np.array([dx, dy, 0.0, float(is_detected)], dtype=np.float32)
        
        half_width = self.window_width // 2
        half_height = self.window_height // 2
        
        x_min_window = int(self.dyn_center_x - half_width)
        y_min_window = int(self.dyn_center_y - half_height)
        x_max_window = int(self.dyn_center_x + half_width)
        y_max_window = int(self.dyn_center_y + half_height)

        cv2.rectangle(frame_with_detections, 
                      (x_min_window, y_min_window), 
                      (x_max_window, y_max_window), 
                      (0, 255, 0), 2) 
        
        return initial_state, frame_with_detections

    def detect_ball(self):
        if self.current_frame is None:
            return 0.0, 0.0, None, False

        detected_boxes, frame_with_detections = get_ball_detection(
            self.detection_model, self.current_frame.copy(), self.transform, self.device
        )
        is_detected = False
        if detected_boxes:
            box = detected_boxes[0]['box']
            x_min, y_min, x_max, y_max = box
            
            ball_x = (x_min + x_max) // 2
            ball_y = (y_min + y_max) // 2
            is_detected = True
        else:
            
            ball_x = self.dyn_center_x 
            ball_y = self.dyn_center_y
            is_detected = False
            
        return ball_x, ball_y, frame_with_detections, is_detected


    def step(self, action):
        
        pan_action = action.flatten()[0]
        tilt_action = 0.0
        
        
        tilt_shift = 0.0 
        
        
        pan_shift = pan_action * (self.W * 0.1 / self.max_action)
        
        self.dyn_center_x += pan_shift
        
        self.dyn_center_y = self.FRAME_CENTER[1] 
        
        
        self.dyn_center_x = np.clip(self.dyn_center_x, self.max_center_shift_x, self.W - self.max_center_shift_x)
        
        
        command = f"P:{pan_action:.2f},T:{0.0:.2f}\n"
        if self.ser:
            
            self.ser.write(command.encode('utf-8'))
        print(f"{command}")
        
        
        ret, next_frame = self.cap.read()
        if not ret:
            
            done = True
            
            next_state = np.zeros(4, dtype=np.float32) 
            reward = 0.0
            frame_with_detections = self.current_frame
            
            return next_state, reward, done, np.zeros(1), frame_with_detections 

        self.current_frame = next_frame
        
        ball_x, ball_y, frame_with_detections, is_detected = self.detect_ball()
        
        
        dx = (ball_x - self.dyn_center_x) / self.W
        dy = (ball_y - self.dyn_center_y) / self.H
        
        
        next_state = np.array([dx, dy, pan_action, float(is_detected)], dtype=np.float32)

        reward = self.reward_system.calculate_reward(dx, dy, pan_action, is_detected)
        self.reward_system.update_prev_action(pan_action)
        
        done = False
        
        
        half_width = self.window_width // 2
        half_height = self.window_height // 2
        
        x_min_window = int(self.dyn_center_x - half_width)
        y_min_window = int(self.dyn_center_y - half_height)
        x_max_window = int(self.dyn_center_x + half_width)
        y_max_window = int(self.dyn_center_y + half_height)

        cv2.rectangle(frame_with_detections, 
                      (x_min_window, y_min_window), 
                      (x_max_window, y_max_window), 
                      (0, 255, 0), 2) 
        
        
        return next_state, reward, done, np.array([pan_action]), frame_with_detections

    def set_current_frame(self, frame):
        self.current_frame = frame

    def get_state(self):
        ball_x, ball_y, frame_with_detections, is_detected = self.detect_ball()
        dx = (ball_x - self.dyn_center_x) / self.W
        dy = (ball_y - self.dyn_center_y) / self.H
        state = np.array([dx, dy, 0.0, float(is_detected)], dtype=np.float32)
        return state, frame_with_detections

    def execute_action(self, action):
        pan_action = action.flatten()[0]
        pan_shift = pan_action * (self.W * 0.1 / self.max_action)
        self.dyn_center_x += pan_shift
        self.dyn_center_x = np.clip(self.dyn_center_x, self.max_center_shift_x, self.W - self.max_center_shift_x)
        command = f"P:{pan_action:.2f},T:{0.0:.2f}\n"
        if self.ser:
            self.ser.write(command.encode('utf-8'))
        print(f"Sent command: {command.strip()}")
