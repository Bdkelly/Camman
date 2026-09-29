"""Process bounded capture frames; UI presentation pulls from a separate mailbox."""

import time

from PyQt5.QtCore import QMutexLocker

from backend.detection import get_ball_detection


def videorun(thread, reader, transform, fps):
    file_period = 1 / fps if fps and 0 < fps < 240 else 1 / 30
    next_frame = next_preview = 0.0
    stats_started = time.monotonic()
    displayed = inferred = 0
    inference_ms = 0.0
    previous_inference = None
    while thread._run_flag:
        thread.load_pending_model()
        thread.send_pending_command()
        with QMutexLocker(thread.mutex):
            inference = thread.inference_active and thread.model is not None
            tracking = thread.agent_active
            model = thread.model
            interval = thread.command_interval
        if inference != previous_inference:
            # Toggling inference must not inherit a long deadline from a low
            # inference FPS cap; preview/control changes take effect immediately.
            next_frame = 0.0
            previous_inference = inference
        now = time.monotonic()
        if now < next_frame:
            # Short waits keep controls responsive even at a low requested FPS.
            thread.msleep(max(1, min(10, int((next_frame - now) * 1000))))
            continue
        packet = reader.read()
        if packet is None:
            if reader.ended:
                if reader.error:
                    raise RuntimeError(f"Capture failed: {reader.error}")
                thread.command_log_signal.emit("Video ended or camera stopped delivering frames.")
                break
            continue
        started = time.monotonic()
        frame = packet.image
        if inference:
            boxes, frame = get_ball_detection(
                model, frame, transform, thread.device, precision=thread.runtime.precision
            )
            inference_ms = (time.monotonic() - started) * 1000
            inferred += 1
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
        now = time.monotonic()
        if now >= next_preview:
            thread.publish_preview(frame)
            displayed += 1
            next_preview = max(next_preview + 1 / thread.runtime.preview_fps, now)
        elapsed = now - stats_started
        if elapsed >= 1:
            thread.stats_signal.emit(
                f"Preview {displayed / elapsed:.1f} FPS | Detect {inferred / elapsed:.1f} FPS | "
                f"Last detection {inference_ms:.0f} ms | "
                f"Frame age {(now - packet.captured_at) * 1000:.0f} ms | "
                f"Capture drops {reader.dropped}"
            )
            displayed = inferred = 0
            stats_started = now
        period = 0.0 if reader.live else file_period
        if inference and thread.runtime.inference_fps:
            period = max(period, 1 / thread.runtime.inference_fps)
        elif reader.live:
            period = 1 / thread.runtime.preview_fps
        # Inference counts toward this budget. On live inputs the reader keeps
        # draining capture while we work/wait; the next iteration gets newest.
        next_frame = started + period
