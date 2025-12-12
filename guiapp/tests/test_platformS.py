import pytest
import asyncio
from unittest.mock import Mock, patch
from PyQt5.QtCore import Qt
from guiapp.platform_screen import WiredWorker, BluetoothWorker, StatusIndicator, PlatformWindow

def test_wired_worker_success(qtbot):
    with patch('platform_screen.find_esp32', return_value="COM3"):
        worker = WiredWorker()
        with qtbot.waitSignal(worker.result_signal) as blocker:
            worker.start()
        assert blocker.args == ["COM3"]

def test_wired_worker_failure(qtbot):
    with patch('platform_screen.find_esp32', return_value=None):
        worker = WiredWorker()
        with qtbot.waitSignal(worker.result_signal) as blocker:
            worker.start()
        assert blocker.args == ["None"]

def test_bluetooth_worker_found(qtbot):
    f = asyncio.Future()
    f.set_result(True)
    
    mock_controller = Mock()
    mock_controller.scan_and_check.return_value = f

    with patch('platform_screen.ESP32Controller', return_value=mock_controller):
        worker = BluetoothWorker()
        with qtbot.waitSignal(worker.status_signal) as blocker:
            worker.start()
        assert blocker.args == [True]

def test_bluetooth_worker_not_found(qtbot):
    f = asyncio.Future()
    f.set_result(False)
    
    mock_controller = Mock()
    mock_controller.scan_and_check.return_value = f

    with patch('platform_screen.ESP32Controller', return_value=mock_controller):
        worker = BluetoothWorker()
        with qtbot.waitSignal(worker.status_signal) as blocker:
            worker.start()
        assert blocker.args == [False]

def test_status_indicator_color(qtbot):
    widget = StatusIndicator(Qt.red)
    qtbot.addWidget(widget)
    assert widget.color == Qt.red
    widget.set_color(Qt.green)
    assert widget.color == Qt.green

def test_platform_window_initialization(qtbot):
    f = asyncio.Future()
    f.set_result(False)
    
    mock_controller = Mock()
    mock_controller.scan_and_check.return_value = f

    with patch('platform_screen.find_esp32', return_value=None), \
         patch('platform_screen.ESP32Controller', return_value=mock_controller):
        
        window = PlatformWindow()
        qtbot.addWidget(window)
        assert window.windowTitle() == "Platform Connectivity"
        assert "Unknown" in window.wired_label.text()
        assert "Scanning" in window.bt_label.text()

def test_wired_scan_button_interaction(qtbot):
    f = asyncio.Future()
    f.set_result(False)
    
    mock_controller = Mock()
    mock_controller.scan_and_check.return_value = f

    with patch('platform_screen.find_esp32', return_value="COM4"), \
         patch('platform_screen.ESP32Controller', return_value=mock_controller):
        
        window = PlatformWindow()
        qtbot.addWidget(window)
        assert window.wired_indicator.color == Qt.red
        
        with qtbot.waitSignal(window.wired_worker.result_signal, timeout=1000):
            qtbot.mouseClick(window.scan_wired_btn, Qt.LeftButton)
            
        assert "Connected (COM4)" in window.wired_label.text()
        assert window.wired_indicator.color == Qt.green

def test_wired_scan_button_fail(qtbot):
    f = asyncio.Future()
    f.set_result(False)
    
    mock_controller = Mock()
    mock_controller.scan_and_check.return_value = f

    with patch('platform_screen.find_esp32', return_value=None), \
         patch('platform_screen.ESP32Controller', return_value=mock_controller):
        
        window = PlatformWindow()
        qtbot.addWidget(window)
        
        with qtbot.waitSignal(window.wired_worker.result_signal, timeout=1000):
            qtbot.mouseClick(window.scan_wired_btn, Qt.LeftButton)
            
        assert "Disconnected" in window.wired_label.text()
        assert window.wired_indicator.color == Qt.red

def test_bluetooth_ui_updates(qtbot):
    f = asyncio.Future()
    f.set_result(True)
    
    mock_controller = Mock()
    mock_controller.scan_and_check.return_value = f

    with patch('platform_screen.find_esp32', return_value=None), \
         patch('platform_screen.ESP32Controller', return_value=mock_controller):
        
        window = PlatformWindow()
        qtbot.addWidget(window)
        
        with qtbot.waitSignal(window.bt_worker.status_signal, timeout=2000):
            pass
            
        assert "Connected/Found" in window.bt_label.text()
        assert window.bt_indicator.color == Qt.green