"""
camController — Gym-style environment for RL-based camera pan control.

``CameraControlEnv`` wraps a video source (``cv2.VideoCapture``) and a
Faster R-CNN detection model into an OpenAI Gym-compatible interface
(``reset`` / ``step``).  At each step the agent provides a continuous pan
action; the environment shifts an internal *dynamic centre point* to
simulate the camera moving, detects the ball in the new frame, and returns
a 4-D state vector along with a reward from the ``RewardSystem``.

State vector: ``[dx, dy, pan_action, is_detected]``

- ``dx`` / ``dy``: normalised ball offset from dynamic centre (−1 to 1).
- ``pan_action``: the action taken this step (provides velocity context).
- ``is_detected``: 1.0 if the ball was found, 0.0 otherwise.

A green rectangle is drawn on every returned frame to visualise the agent's
current tracking window.
"""

#Agent Calls
try:
    from ballfind import get_ball_detection
    from reward import RewardSystem
except:
    from .ballfind import get_ball_detection
    from .reward import RewardSystem
#Standard Calls
import cv2
import numpy as np
import torch
import serial


class CameraControlEnv:
    """Gym-style camera control environment backed by a video file.

    Each episode runs through a video file from frame 0. The environment
    simulates camera pan by shifting an internal dynamic centre point rather
    than physically moving the camera — this allows offline training on
    pre-recorded footage. During deployment (``execute_action``), real pan
    commands are sent to the ESP32 via serial.

    Args:
        cap (cv2.VideoCapture): Opened video capture object (file or camera).
        detection_model (torch.nn.Module): Faster R-CNN model in eval mode.
        transform (albumentations.Compose): Image preprocessing transform.
        device (torch.device): Device for model inference.
        frame_center_x (int): Horizontal centre of the frame in pixels.
        frame_center_y (int): Vertical centre of the frame in pixels.
        max_action (float): Maximum pan angle magnitude passed to ``ballfind``.
        reward_system (RewardSystem): Configured reward function instance.

    Attributes:
        W (int): Frame width in pixels.
        H (int): Frame height in pixels.
        FRAME_CENTER (tuple[int, int]): Fixed (cx, cy) pixel coordinates.
        dyn_center_x (float): Current dynamic centre x (shifts with pan actions).
        dyn_center_y (float): Current dynamic centre y (fixed at frame centre).
        window_width (int): Width of the agent's tracking window (25% of W).
        window_height (int): Height of the tracking window (full frame height).
        ser (serial.Serial | None): Optional serial connection for hardware commands.
        current_frame (np.ndarray | None): Most recently read video frame.
    """

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
        """Start a new episode from frame 0.

        Seeks the video back to the beginning, reads the first frame,
        resets the reward system's history, and re-centres the dynamic
        tracking window. Detects the ball in the first frame to compute
        the initial state.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - ``initial_state``: 4-D state vector ``[dx, dy, 0.0, is_detected]``.
                - ``frame_with_detections``: Annotated first frame with bounding box
                  and green tracking window overlay.
        """
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
        """Run ball detection on the current frame.

        Delegates to ``get_ball_detection`` from ``ballfind``. When no ball
        is found the dynamic centre is returned as the ball position so
        downstream state computation remains valid.

        Returns:
            tuple[float, float, np.ndarray | None, bool]:
                - ``ball_x``: Ball centre x in pixels (or dyn_center_x if undetected).
                - ``ball_y``: Ball centre y in pixels (or dyn_center_y if undetected).
                - ``frame_with_detections``: Annotated frame (or None if no frame loaded).
                - ``is_detected``: True if at least one box passed the threshold.
        """
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
        """Apply a pan action and advance the environment by one frame.

        Shifts the dynamic centre proportional to ``pan_action``, reads the
        next video frame, detects the ball, computes the new state and reward,
        and optionally sends the command to the ESP32 via serial.

        If the video source is exhausted (``cap.read()`` returns False),
        the episode is terminated with a zero reward and a zero state.

        Args:
            action (np.ndarray): 1-D or scalar array containing the pan angle.

        Returns:
            tuple[np.ndarray, float, bool, np.ndarray, np.ndarray]:
                - ``next_state``: 4-D state ``[dx, dy, pan_action, is_detected]``.
                - ``reward``: Scalar reward from the reward system.
                - ``done``: True if the video is exhausted.
                - ``action_taken``: 1-D array with the actual pan command sent.
                - ``frame_with_detections``: Annotated frame with tracking overlay.
        """
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
        """Inject an external frame as the current frame.

        Used when the video source is managed externally (e.g. the GUI
        ``VideoThread``) and frames are pushed in rather than pulled via
        ``cap.read()``.

        Args:
            frame (np.ndarray): BGR image to use as the current frame.
        """
        self.current_frame = frame

    def get_state(self):
        """Compute the current state without advancing the video.

        Runs ball detection on the current frame and returns a state with
        ``pan_action = 0.0`` (no action was taken).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - ``state``: 4-D state vector ``[dx, dy, 0.0, is_detected]``.
                - ``frame_with_detections``: Annotated current frame.
        """
        ball_x, ball_y, frame_with_detections, is_detected = self.detect_ball()
        dx = (ball_x - self.dyn_center_x) / self.W
        dy = (ball_y - self.dyn_center_y) / self.H
        state = np.array([dx, dy, 0.0, float(is_detected)], dtype=np.float32)
        return state, frame_with_detections

    def execute_action(self, action):
        """Apply a pan action and send the serial command (no frame advance).

        Designed for real-time inference in the GUI where frames are
        provided externally. Updates the dynamic centre and sends the
        formatted pan command to the ESP32 if a serial connection is open.

        Args:
            action (np.ndarray): 1-D or scalar array containing the pan angle.
        """
        pan_action = action.flatten()[0]
        pan_shift = pan_action * (self.W * 0.1 / self.max_action)
        self.dyn_center_x += pan_shift
        self.dyn_center_x = np.clip(self.dyn_center_x, self.max_center_shift_x, self.W - self.max_center_shift_x)
        command = f"P:{pan_action:.2f},T:{0.0:.2f}\n"
        if self.ser:
            self.ser.write(command.encode('utf-8'))
        print(f"Sent command: {command.strip()}")
