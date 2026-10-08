from unittest.mock import Mock

import numpy as np
import pytest
from PyQt5.QtGui import QImage

from frontend.threads.video_threads import VideoThread


@pytest.fixture
def video_thread(qapp):
    thread = VideoThread(device="cpu")
    yield thread
    thread.stop()


def test_initialization(video_thread):
    assert video_thread._run_flag
    assert not video_thread.inference_active
    assert not video_thread.agent_active
    assert video_thread.command_interval == 0.1


def test_video_eof_releases_capture_and_serial(video_thread, mocker):
    cap, serial = Mock(), Mock()
    serial.write.side_effect = len
    mocker.patch("frontend.threads.video_threads.verify_velocity_controller")
    cap.isOpened.return_value = True
    cap.get.return_value = 30
    cap.read.side_effect = [(True, np.zeros((20, 30, 3), dtype=np.uint8)), (False, None)]
    mocker.patch("frontend.threads.video_threads.cv2.VideoCapture", return_value=cap)
    mocker.patch(
        "frontend.threads.video_threads.models_directory",
        return_value=Mock(glob=Mock(return_value=[])),
    )
    mocker.patch("frontend.threads.video_threads.open_connection", return_value=serial)
    video_thread.run()
    assert isinstance(video_thread.take_preview(), QImage)
    assert video_thread.take_preview() is None
    cap.release.assert_called_once()
    serial.close.assert_called_once()
    assert not video_thread._run_flag
    assert video_thread.ser is None


def test_camera_failure_releases_capture(video_thread, mocker):
    cap = Mock()
    cap.isOpened.return_value = False
    mocker.patch("frontend.threads.video_threads.cv2.VideoCapture", return_value=cap)
    logs = []
    video_thread.command_log_signal.connect(logs.append)
    video_thread.run()
    assert any("Could not open video" in log for log in logs)
    cap.release.assert_called_once()
    assert not video_thread._run_flag


def test_inference_exception_closes_resources(video_thread, mocker):
    cap, serial = Mock(), Mock()
    serial.write.side_effect = len
    mocker.patch("frontend.threads.video_threads.verify_velocity_controller")
    cap.isOpened.return_value = True
    cap.get.return_value = 30
    cap.read.return_value = (False, None)
    mocker.patch("frontend.threads.video_threads.cv2.VideoCapture", return_value=cap)
    mocker.patch(
        "frontend.threads.video_threads.models_directory",
        return_value=Mock(glob=Mock(return_value=[])),
    )
    mocker.patch("frontend.threads.video_threads.open_connection", return_value=serial)
    mocker.patch(
        "frontend.threads.video_threads.videorun", side_effect=RuntimeError("inference failed")
    )
    video_thread.run()
    cap.release.assert_called_once()
    serial.close.assert_called_once()


def test_update_model_runs_in_worker(video_thread, mocker):
    model = Mock()
    loader = mocker.patch("frontend.threads.video_threads.load_model_from_path", return_value=model)
    video_thread.update_model("new.pth")
    loader.assert_not_called()
    video_thread.load_pending_model()
    assert video_thread.model is model
    loader.assert_called_once()


def test_failed_update_retains_previous_model(video_thread, mocker):
    previous = video_thread.model = Mock()
    mocker.patch(
        "frontend.threads.video_threads.load_model_from_path", side_effect=ValueError("bad weights")
    )
    video_thread.update_model("bad.pth")
    video_thread.load_pending_model()
    assert video_thread.model is previous


def test_controls(video_thread):
    video_thread.toggle_inference(True)
    video_thread.toggle_agent(True)
    video_thread.set_command_interval(2.5)
    assert video_thread.inference_active and video_thread.agent_active
    assert video_thread.command_interval == 2.5
    video_thread.toggle_inference(False)
    video_thread.toggle_agent(False)
    assert not video_thread.inference_active and not video_thread.agent_active


def test_convert_cv_qt_owns_image(video_thread):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    image = video_thread._convert_cv_qt(frame)
    frame[:] = 255
    assert isinstance(image, QImage)
    assert image.height() == image.width() == 480
    assert image.pixelColor(0, 0).red() == 0


def test_stop(video_thread, mocker):
    wait = mocker.patch.object(video_thread, "wait")
    video_thread.stop()
    assert not video_thread._run_flag
    wait.assert_called_once()


def test_preview_mailbox_retains_only_newest_image(video_thread):
    video_thread.publish_preview(np.zeros((1080, 1920, 3), dtype=np.uint8))
    video_thread.publish_preview(np.full((1080, 1920, 3), 255, dtype=np.uint8))
    image = video_thread.take_preview()
    assert (image.width(), image.height()) == (640, 360)
    assert image.pixelColor(0, 0).red() == 255
    assert video_thread.take_preview() is None


def test_manual_commands_are_coalesced_and_sent_by_worker(video_thread):
    video_thread.ser = Mock(write=Mock(side_effect=len))
    video_thread.request_manual_command("Left")
    video_thread.request_manual_command("Right")
    video_thread.ser.write.assert_not_called()
    video_thread.send_pending_command()
    video_thread.send_pending_command()
    video_thread.ser.write.assert_called_once_with(b"V:0.2500\n")


def test_file_inference_preserves_every_frame(video_thread, mocker):
    cap = Mock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30
    frames = [np.full((20, 30, 3), value, dtype=np.uint8) for value in range(3)]
    cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    mocker.patch("frontend.threads.video_threads.cv2.VideoCapture", return_value=cap)
    mocker.patch(
        "frontend.threads.video_threads.models_directory",
        return_value=Mock(glob=Mock(return_value=[])),
    )
    detector = mocker.patch(
        "frontend.video.get_ball_detection",
        side_effect=lambda model, frame, *args, **kw: ([], frame),
    )
    video_thread.model = Mock()
    video_thread.inference_active = True
    video_thread.live = False
    video_thread.run()
    assert detector.call_count == 3
    assert [int(call.args[1][0, 0, 0]) for call in detector.call_args_list] == [0, 1, 2]
    assert all(call.kwargs["precision"] == "fp32" for call in detector.call_args_list)
    cap.release.assert_called_once()


def test_stopping_inference_does_not_stall_preview_until_rate_deadline(video_thread, mocker):
    from dataclasses import replace

    from backend.capture import CapturedFrame
    from backend.detection import FrameTransform
    from frontend.video import videorun

    clock = [1.0]
    mocker.patch("frontend.video.time.monotonic", side_effect=lambda: clock[0])
    mocker.patch.object(
        video_thread, "msleep", side_effect=lambda ms: clock.__setitem__(0, clock[0] + ms / 1000)
    )
    video_thread.runtime = replace(video_thread.runtime, inference_fps=0.1)
    video_thread.model = Mock()
    video_thread.inference_active = True
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    reader = Mock(live=False, ended=True, error=None, dropped=0)
    reader.read.side_effect = [CapturedFrame(frame, 1), CapturedFrame(frame, 1), None]

    def detect(*args, **kwargs):
        video_thread.toggle_inference(False)
        return [], frame

    mocker.patch("frontend.video.get_ball_detection", side_effect=detect)
    videorun(video_thread, reader, FrameTransform(), 30)
    assert clock[0] < 1.1  # Preview resumes without waiting the ten-second cap.


def test_incompatible_controller_disables_motion(video_thread, mocker):
    cap = Mock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30
    cap.read.return_value = (False, None)
    connection = Mock(write=Mock(side_effect=len))
    mocker.patch("frontend.threads.video_threads.open_capture", return_value=cap)
    mocker.patch("frontend.threads.video_threads.open_connection", return_value=connection)
    mocker.patch(
        "frontend.threads.video_threads.verify_velocity_controller",
        side_effect=RuntimeError("wrong protocol"),
    )
    mocker.patch(
        "frontend.threads.video_threads.models_directory",
        return_value=Mock(glob=Mock(return_value=[])),
    )
    video_thread.run()
    assert not video_thread.tracking_available
    video_thread.toggle_agent(True)
    assert not video_thread.agent_active
    connection.close.assert_called_once()


def test_disabling_tracking_during_detection_sends_stop(video_thread, mocker):
    from backend.capture import CapturedFrame
    from backend.detection import FrameTransform
    from frontend.video import videorun

    mocker.patch("frontend.video.time.monotonic", return_value=10)
    video_thread.inference_active = video_thread.agent_active = True
    video_thread.model = Mock()
    video_thread.ser = Mock(write=Mock(side_effect=len))
    video_thread.controller.prev_action = 0.5
    video_thread.controller.observations.last_seen = 10
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    reader = Mock(live=True, ended=True, error=None)
    reader.read.side_effect = [CapturedFrame(frame, 10), None]
    video_thread.runtime = __import__("dataclasses").replace(video_thread.runtime, inference_fps=0)

    def detect(*args, **kwargs):
        video_thread.toggle_agent(False)
        video_thread._run_flag = False
        return [{"box": (20, 5, 28, 15)}], frame

    mocker.patch("frontend.video.get_ball_detection", side_effect=detect)
    videorun(video_thread, reader, FrameTransform(), 30)
    video_thread.ser.write.assert_called_once_with(b"Stop\n")
