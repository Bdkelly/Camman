from unittest.mock import Mock

import numpy as np
import pytest

from backend.capture import CapturedFrame
from backend.runtime import resolve_runtime
from backend.tools.track import run_tracking
from backend.tracking import TrackingController


@pytest.mark.parametrize("failure", [False, True])
def test_headless_stream_stops_on_eof_or_inference_error(mocker, failure):
    clock = [10.0]
    mocker.patch("backend.tools.track.time.monotonic", side_effect=lambda: clock[0])
    frame = np.zeros((100, 100, 3), np.uint8)
    reader = Mock(live=True, ended=True, error=None)
    reader.read.side_effect = [CapturedFrame(frame, 10), CapturedFrame(frame, 10.1), None]
    detections = [{"box": (75, 45, 85, 55)}]
    calls = [0]

    def detect(*args, **kwargs):
        calls[0] += 1
        clock[0] += 0.1
        if failure and calls[0] == 2:
            raise RuntimeError("detector failure")
        assert kwargs["draw"] is False
        return detections, frame

    mocker.patch("backend.tools.track.get_ball_detection", side_effect=detect)
    ser = Mock(write=Mock(side_effect=len))
    controller = TrackingController()
    kwargs = dict(ser=ser, log=None)
    if failure:
        with pytest.raises(RuntimeError, match="detector failure"):
            run_tracking(reader, Mock(), resolve_runtime(device="cpu"), controller, **kwargs)
    else:
        assert (
            run_tracking(reader, Mock(), resolve_runtime(device="cpu"), controller, **kwargs) == 2
        )
    assert ser.write.call_args_list[0].args == (b"V:0.9000\n",)
    ser.write.assert_called_with(b"Stop\n")
    assert controller.prev_action == 0
