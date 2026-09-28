"""Serial transport with plain callbacks, usable without Qt."""

import serial
import serial.tools.list_ports

from backend.hardware.serial_validation import valid_serial


def scan_port(port):
    return valid_serial(port)


def find_esp32():
    for port in serial.tools.list_ports.comports():
        if scan_port(port):
            return port.device
    return None


def open_connection(port):
    """Connect to an explicit port, or probe when the caller requests 'auto'."""
    if port == "auto":
        port = find_esp32()
    return serial.Serial(port, 115200, timeout=1, write_timeout=1) if port else None


def send_agent_command(ser, command, log=None):
    if ser is None:
        return
    text = command.strip()
    ser.write((text + "\n").encode("utf-8"))
    if log is not None:
        log(text)


def move_left(ser, log=None):
    send_agent_command(ser, "Left", log)


def move_right(ser, log=None):
    send_agent_command(ser, "Right", log)
