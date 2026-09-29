from unittest.mock import Mock

import numpy as np
import pytest

from backend.runtime import resolve_runtime
from backend.tools.benchmark import measure


def test_benchmark_excludes_warmup_and_reports_early_eof(mocker):
    cap = Mock()
    cap.read.side_effect = [(True, np.zeros((16, 16, 3), dtype=np.uint8))] * 4 + [(False, None)]
    detector = mocker.patch("backend.tools.benchmark.get_ball_detection", return_value=([{}], None))
    report = measure(Mock(), cap, resolve_runtime(device="cpu"), frames=10, warmup=1)
    assert report["measured_frames"] == report["frames_with_detection"] == 3
    assert report["requested_frames"] == 10
    assert report["detection_ms"]["p95"] >= report["detection_ms"]["p50"] > 0
    assert report["sequential_throughput_fps"] > 0
    assert report["peak_cuda_allocated_mib"] is None
    assert detector.call_count == 4


def test_benchmark_does_not_report_empty_measurements():
    cap = Mock()
    cap.read.return_value = False, None
    with pytest.raises(ValueError, match="No measured frames"):
        measure(Mock(), cap, resolve_runtime(device="cpu"))
