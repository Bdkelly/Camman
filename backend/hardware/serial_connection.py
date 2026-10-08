"""Serial transport with plain callbacks, usable without Qt."""

import time

import serial
import serial.tools.list_ports

from backend.hardware.protocol import PROTOCOL_GREETING
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
    if not port:
        return None
    factory = serial.serial_for_url if "://" in port else serial.Serial
    return factory(port, 115200, timeout=0.2, write_timeout=0.2)


def send_agent_command(ser, command, log=None):
    if ser is None:
        return
    text = command.strip()
    if not text or "\n" in text or "\r" in text:
        raise ValueError("A controller command must be one nonempty line")
    payload = (text + "\n").encode("ascii")
    written = ser.write(payload)
    if written != len(payload):
        raise serial.SerialTimeoutException("Incomplete controller command write")
    if log is not None:
        log(text)


def verify_velocity_controller(ser, *, timeout=3.0):
    """Wait through USB reset and require explicit support for velocity commands."""
    if ser is None:
        return
    deadline = time.monotonic() + timeout
    ser.reset_input_buffer()
    send_agent_command(ser, "Stop")
    while time.monotonic() < deadline:
        send_agent_command(ser, "HELLO")
        response = ser.readline().decode("ascii", errors="replace").strip()
        if response == PROTOCOL_GREETING:
            return
    raise RuntimeError(
        "Controller did not identify as CAMMAN/1 VELOCITY. Install the controller firmware "
        "in backend/firmware/controller or implement that protocol on your microcontroller."
    )


def move_left(ser, log=None):
    send_agent_command(ser, "V:-0.2500", log)


def move_right(ser, log=None):
    send_agent_command(ser, "V:0.2500", log)
