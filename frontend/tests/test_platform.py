from unittest.mock import AsyncMock

import pytest
from PyQt5.QtCore import Qt

from frontend.platform_screen import BluetoothWorker, PlatformWindow, StatusIndicator, WiredWorker


@pytest.mark.parametrize("port", ["COM3", None])
def test_wired_worker(qtbot, mocker, port):
    mocker.patch("frontend.platform_screen.find_esp32", return_value=port)
    worker = WiredWorker()
    with qtbot.waitSignal(worker.result_signal) as blocker:
        worker.start()
    worker.wait()
    assert blocker.args == [port or "None"]


@pytest.mark.parametrize("found", [True, False])
def test_bluetooth_worker(qtbot, mocker, found):
    mocker.patch(
        "frontend.platform_screen.ESP32Controller.scan_and_check", new=AsyncMock(return_value=found)
    )
    worker = BluetoothWorker()
    with qtbot.waitSignal(worker.status_signal) as blocker:
        worker.start()
    worker.wait()
    assert blocker.args == [found]


def test_missing_bluetooth_adapter_is_reported(qtbot, mocker):
    mocker.patch(
        "frontend.platform_screen.ESP32Controller.scan_and_check",
        new=AsyncMock(side_effect=RuntimeError("No adapter")),
    )
    worker = BluetoothWorker()
    with qtbot.waitSignal(worker.status_signal) as blocker:
        worker.start()
    worker.wait()
    assert blocker.args == [False]


def test_status_indicator_color(qtbot):
    widget = StatusIndicator(Qt.red)
    qtbot.addWidget(widget)
    widget.set_color(Qt.green)
    assert widget.color == Qt.green


@pytest.mark.parametrize("port", ["COM4", None])
def test_platform_scan(qtbot, mocker, port):
    mocker.patch("frontend.platform_screen.find_esp32", return_value=port)
    mocker.patch(
        "frontend.platform_screen.ESP32Controller.scan_and_check", new=AsyncMock(return_value=False)
    )
    window = PlatformWindow()
    qtbot.addWidget(window)
    window.bt_worker.wait()
    with qtbot.waitSignal(window.wired_worker.result_signal):
        qtbot.mouseClick(window.scan_wired_btn, Qt.LeftButton)
    window.wired_worker.wait()
    assert window.wired_indicator.color == (Qt.green if port else Qt.red)
    assert (f"Connected ({port})" if port else "Disconnected") in window.wired_label.text()
    window.close()
