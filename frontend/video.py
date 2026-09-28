"""Qt video presentation adapter; detection and control live in backend/."""

import time

import cv2
from PyQt5.QtCore import QMutexLocker

from backend.detection import get_ball_detection


def videorun(thread, cap, transform):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_period = 1 / fps if fps and 0 < fps < 240 else 1 / 30
    while thread._run_flag:
        started = time.monotonic()
        thread.load_pending_model()
        ok, frame = cap.read()
        if not ok:
            thread.command_log_signal.emit("Video ended or camera stopped delivering frames.")
            break
        with QMutexLocker(thread.mutex):
            inference = thread.inference_active
            tracking = thread.agent_active
            model = thread.model
            interval = thread.command_interval
        if inference and model is not None:
            boxes, frame = get_ball_detection(model, frame, transform, thread.device)
            if tracking:
                height, width = frame.shape[:2]
                thread.controller.update(
                    boxes,
                    thread.ser,
                    width,
                    height,
                    interval,
                    agent=thread.agent,
                    log=thread.command_log_signal.emit,
                )
        thread.change_pixmap_signal.emit(thread._convert_cv_qt(frame))
        # Pace playback at the video's rate; inference time counts toward the frame budget.
        remaining = frame_period - (time.monotonic() - started)
        if remaining > 0:
            thread.msleep(max(1, int(remaining * 1000)))
