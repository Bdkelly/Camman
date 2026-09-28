"""Camera control state and rate limiting, independent of the UI toolkit."""

import time

import numpy as np

from backend.hardware.serial_connection import move_left, move_right, send_agent_command


class TrackingController:
    def __init__(self):
        self.last_command_time = 0.0
        self.prev_action = 0.0

    def update(self, boxes, ser, width, height, interval, *, agent=None, log=None):
        now = time.monotonic()
        if ser is None or now - self.last_command_time < interval:
            return None
        if agent is not None:
            dx = dy = detected = 0.0
            if boxes:
                x1, y1, x2, y2 = boxes[0]["box"]
                dx = ((x1 + x2) / 2 - width / 2) / width
                dy = ((y1 + y2) / 2 - height / 2) / height
                detected = 1.0
            state = np.array([dx, dy, self.prev_action, detected], dtype=np.float32)
            action = float(agent.choose_action(state)[0])
            send_agent_command(ser, f"P:{action:.2f},T:0.00", log)
            self.prev_action = action
            self.last_command_time = now
            return action
        if boxes:
            x1, _, x2, _ = boxes[0]["box"]
            center = (x1 + x2) / 2
            if center < width / 2 - 50:
                move_left(ser, log)
            elif center > width / 2 + 50:
                move_right(ser, log)
            else:
                send_agent_command(ser, "Stop", log)
            self.last_command_time = now
        return None
