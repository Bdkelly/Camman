import pytest
from unittest.mock import Mock, patch
from guiapp.utils.ser_con import move_left, move_right, find_esp32, scan_port, set_command_signal

def test_move_left():
    ser = Mock()
    mock_signal = Mock()
    set_command_signal(mock_signal)
    
    move_left(ser)
    
    ser.write.assert_called_with(b"Left\n")
    mock_signal.emit.assert_called_with("Left")

def test_move_left_no_ser():
    ser = None
    move_left(ser)

def test_move_right():
    ser = Mock()
    mock_signal = Mock()
    set_command_signal(mock_signal)
    
    move_right(ser)
    
    ser.write.assert_called_with(b"Right\n")
    mock_signal.emit.assert_called_with("Right")

def test_scan_port_valid():
    port = Mock()
    with patch('guiapp.utils.ser_con.valid_serial', return_value="Move: 0.99 (Right)"):
        assert scan_port(port) is True

def test_scan_port_invalid():
    port = Mock()
    with patch('guiapp.utils.ser_con.valid_serial', return_value="X"):
        assert scan_port(port) is False

def test_find_esp32_success():
    port = Mock()
    port.device = "/dev/ttyUSB0"
    port.description = "ESP32"
    
    with patch('serial.tools.list_ports.comports', return_value=[port]), \
         patch('guiapp.utils.ser_con.scan_port', return_value=True):
        
        result = find_esp32()
        assert "Serial Connection Valid" in result

def test_find_esp32_fail():
    port = Mock()
    with patch('serial.tools.list_ports.comports', return_value=[port]), \
         patch('guiapp.utils.ser_con.scan_port', return_value=False):
        
        result = find_esp32()
        assert result == "No Valid Connection"