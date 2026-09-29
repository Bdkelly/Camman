import threading
from unittest.mock import Mock

import cv2
import pytest

from backend.capture import FrameReader, is_live_source, open_capture


def test_live_reader_discards_backlog_and_delivers_last_frame_before_eof():
    cap = Mock()
    cap.read.side_effect = [(True, value) for value in range(5)] + [(False, None)]
    reader = FrameReader(cap, live=True).start()
    reader._thread.join(1)
    assert not reader.ended  # Last captured frame is still deliverable.
    assert reader.read().image == 4
    assert reader.dropped == 4
    assert reader.ended
    assert reader.close()
    cap.release.assert_called_once()


def test_file_reader_preserves_order_with_backpressure():
    cap = Mock()
    ready = threading.Event()
    frames = iter([(True, value) for value in range(5)] + [(False, None)])

    def read():
        ready.set()
        return next(frames)

    cap.read.side_effect = read
    reader = FrameReader(cap, live=False).start()
    try:
        assert ready.wait(1)
        assert cap.read.call_count == 1
        assert [reader.read(timeout=1).image for _ in range(5)] == list(range(5))
        assert reader.read(timeout=1) is None
        assert reader.ended and reader.dropped == 0
    finally:
        assert reader.close()
    cap.release.assert_called_once()


def test_close_never_releases_while_native_read_is_running():
    reading, unblock = threading.Event(), threading.Event()
    cap = Mock()

    def read():
        reading.set()
        unblock.wait(2)
        return False, None

    cap.read.side_effect = read
    reader = FrameReader(cap, live=True).start()
    try:
        assert reading.wait(1)
        assert not reader.close(timeout=0.01)
        cap.release.assert_not_called()
    finally:
        unblock.set()
        assert reader.close(timeout=1)
    cap.release.assert_called_once()


def test_capture_error_releases_device_and_reaches_consumer():
    cap = Mock()
    cap.read.side_effect = RuntimeError("disconnected")
    reader = FrameReader(cap, live=True).start()
    assert reader.read(timeout=1) is None
    assert reader.error == "disconnected" and reader.ended
    assert reader.close()
    cap.release.assert_called_once()


def test_release_error_does_not_leave_consumer_waiting_forever():
    cap = Mock()
    cap.read.return_value = (False, None)
    cap.release.side_effect = RuntimeError("driver failure")
    reader = FrameReader(cap, live=True).start()
    assert reader.read(timeout=1) is None
    assert reader.ended and "driver failure" in reader.error
    assert reader.close()


def test_gstreamer_requires_native_build_and_passes_timeout_parameters(mocker):
    factory = mocker.patch("backend.capture.cv2.VideoCapture")
    support = mocker.patch("backend.capture.has_gstreamer", return_value=False)
    with pytest.raises(RuntimeError, match="no GStreamer"):
        open_capture("pipeline", backend="gstreamer")
    factory.assert_not_called()
    support.return_value = True
    cap = open_capture("pipeline", backend="gstreamer", live=True)
    assert factory.call_args.args == (
        "pipeline",
        cv2.CAP_GSTREAMER,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000],
    )
    cap.set.assert_called_once_with(cv2.CAP_PROP_BUFFERSIZE, 1)


@pytest.mark.parametrize(
    "source,mode,backend,live",
    [
        (0, "auto", "auto", True),
        ("/dev/video0", "auto", "v4l2", True),
        ("rtsp://camera/game", "auto", "ffmpeg", True),
        ("game.mp4", "auto", "auto", False),
        ("videotestsrc ! appsink", "auto", "gstreamer", True),
        ("filesrc location=game.mp4 ! appsink", "file", "gstreamer", False),
        ("game.mp4", "auto", "gstreamer", False),
    ],
)
def test_source_mode_is_explicit_for_file_pipelines(source, mode, backend, live):
    assert is_live_source(source, mode, backend) is live
