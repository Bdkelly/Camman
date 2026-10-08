from unittest.mock import Mock

import pytest
import serial

from backend.hardware.serial_connection import (
    find_esp32,
    move_left,
    move_right,
    scan_port,
    send_agent_command,
)
from backend.hardware.serial_validation import valid_serial


@pytest.mark.parametrize("command,wire", [(move_left, b"V:-0.2500\n"), (move_right, b"V:0.2500\n")])
def test_manual_commands_are_framed(command, wire):
    connection, log = Mock(), Mock()
    connection.write.side_effect = len
    command(connection, log)
    connection.write.assert_called_once_with(wire)
    log.assert_called_once_with(wire.decode().strip())


def test_no_serial_is_optional():
    log = Mock()
    move_left(None, log)
    move_right(None, log)
    log.assert_not_called()


def test_agent_command_has_one_newline():
    connection = Mock()
    connection.write.side_effect = len
    send_agent_command(connection, "V:0.5000\n")
    connection.write.assert_called_once_with(b"V:0.5000\n")


@pytest.mark.parametrize("valid", [True, False])
def test_scan_port(mocker, valid):
    mocker.patch("backend.hardware.serial_connection.valid_serial", return_value=valid)
    assert scan_port(Mock()) is valid


def test_find_returns_port_path_not_status_text(mocker):
    port = Mock(device="COM3")
    mocker.patch("serial.tools.list_ports.comports", return_value=[port])
    probe = mocker.patch("backend.hardware.serial_connection.scan_port", return_value=True)
    assert find_esp32() == "COM3"
    probe.return_value = False
    assert find_esp32() is None


def test_probe_closes_connection_and_only_sends_stop(mocker):
    connection = mocker.MagicMock()
    connection.__enter__.return_value = connection
    connection.readline.return_value = b"Stopping\n"
    mocker.patch("backend.hardware.serial_validation.serial.Serial", return_value=connection)
    assert valid_serial(Mock(device="COM3"))
    connection.write.assert_called_once_with(b"Stop\n")
    connection.__exit__.assert_called_once()


def test_unavailable_port_does_not_abort_scan(mocker):
    mocker.patch(
        "backend.hardware.serial_validation.serial.Serial",
        side_effect=serial.SerialException("busy"),
    )
    assert valid_serial(Mock(device="COM3")) is False


def test_partial_write_and_multiple_lines_rejected():
    with pytest.raises(serial.SerialTimeoutException):
        send_agent_command(Mock(write=Mock(return_value=1)), "V:0.2500")
    connection = Mock()
    with pytest.raises(ValueError):
        send_agent_command(connection, "V:0.25\nV:1")
    connection.write.assert_not_called()


def test_controller_handshake_requires_explicit_velocity_support(mocker):
    from backend.hardware.serial_connection import verify_velocity_controller

    connection = Mock(write=Mock(side_effect=len))
    connection.readline.side_effect = [b"Stopping\n", b"CAMMAN/1 VELOCITY\n"]
    verify_velocity_controller(connection)
    assert connection.write.call_args_list[0].args[0] == b"Stop\n"
    clock = mocker.patch(
        "backend.hardware.serial_connection.time.monotonic", side_effect=[0, 0.1, 4]
    )
    connection.readline.side_effect = [b"legacy firmware\n"]
    with pytest.raises(RuntimeError, match="CAMMAN/1"):
        verify_velocity_controller(connection)
    assert clock.call_count == 3
