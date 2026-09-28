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
    assert video_thread.command_interval == 1.0


def test_video_eof_releases_capture_and_serial(video_thread, mocker):
    cap, serial = Mock(), Mock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30
    cap.read.side_effect = [(True, np.zeros((20, 30, 3), dtype=np.uint8)), (False, None)]
    mocker.patch("frontend.threads.video_threads.cv2.VideoCapture", return_value=cap)
    mocker.patch(
        "frontend.threads.video_threads.models_directory",
        return_value=Mock(glob=Mock(return_value=[])),
    )
    mocker.patch("frontend.threads.video_threads.open_connection", return_value=serial)
    frames = []
    video_thread.change_pixmap_signal.connect(frames.append)
    video_thread.run()
    assert len(frames) == 1
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
    cap.isOpened.return_value = True
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
