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


@pytest.mark.parametrize("command,wire", [(move_left, b"Left\n"), (move_right, b"Right\n")])
def test_manual_commands_are_framed(command, wire):
    connection, log = Mock(), Mock()
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
    send_agent_command(connection, "P:0.5,T:0.00\n")
    connection.write.assert_called_once_with(b"P:0.5,T:0.00\n")


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
