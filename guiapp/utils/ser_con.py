"""
ser_con — Serial communication helpers for the ESP32 camera platform.

Provides functions to scan for and validate an ESP32 serial port, send
agent-generated pan commands, and send fixed manual pan commands.

A module-level signal reference (``_command_signal_ref``) is set once via
``set_command_signal`` so that all helpers can emit log messages to the GUI
without passing the signal through every call.

Command protocol (sent as UTF-8 strings):

- ``"P:<value>"``  — set pan angle, e.g. ``"P:2.35"``
- ``"P:0.5"``      — fixed left pan step (``move_left``)
- ``"P:1.5"``      — fixed right pan step (``move_right``)
- ``"Stop\\n"``    — halt movement (sent directly in ``vidpro.track_control``)
"""

import serial
import serial.tools.list_ports
import time
from guiapp.utils.ser_val import valid_serial

_command_signal_ref = None


def set_command_signal(signal):
    """Register a PyQt signal for emitting command log messages.

    Must be called once before any command function is used so that serial
    actions are reflected in the GUI log display.

    Args:
        signal (pyqtSignal): A ``pyqtSignal(str)`` instance, typically
            ``VideoThread.command_log_signal``.
    """
    global _command_signal_ref
    _command_signal_ref = signal


def scan_port(port):
    """Check whether a serial port is a valid ESP32 connection.

    Delegates to ``valid_serial`` which attempts handshake validation.

    Args:
        port: A ``serial.tools.list_ports.ListPortInfo`` object.

    Returns:
        bool: True if the port is valid, False otherwise.
    """
    result = valid_serial(port)
    if result == "X":
        return False
    else:
        return True


def send_agent_command(ser, command):
    """Send a formatted agent command string over serial and log it.

    Args:
        ser (serial.Serial | None): Open serial connection to the ESP32.
        command (str): Command string, e.g. ``"P:2.35"``.
    """
    if ser:
        if _command_signal_ref:
            _command_signal_ref.emit(command.strip())
        ser.write(command.encode('utf-8'))
        print(f"Sent command: {command.strip()}")
    else:
        print("Serial not connected: Agent Command")


def find_esp32():
    """Scan all available serial ports and return the first valid ESP32 port.

    Returns:
        str: A success message string if a valid port is found, or
            ``"Connection"`` if none are found or an error occurs.
    """
    ports = serial.tools.list_ports.comports()
    print(f"Scanning {len(ports)} serial ports...")
    try:
        for port in ports:
            print(f"Checking port: {port.device} - {port.description}")
            if scan_port(port):
                return "Serial Connection Valid: {port.name}"
        return "Connection"
    except:
        return "Connection"


def move_left(ser):
    """Send a fixed left-pan command to the ESP32.

    Args:
        ser (serial.Serial | None): Open serial connection, or None.
    """
    command = "P:0.5"
    if ser:
        if _command_signal_ref:
            _command_signal_ref.emit("Left")
        ser.write(command.encode('utf-8'))
        print("Sent command: Left")
    else:
        print("Serial not connected: Left")


def move_right(ser):
    """Send a fixed right-pan command to the ESP32.

    Args:
        ser (serial.Serial | None): Open serial connection, or None.
    """
    command = "P:1.5"
    if ser:
        if _command_signal_ref:
            _command_signal_ref.emit("Right")
        ser.write(command.encode('utf-8'))
        print("Sent command: Right")
    else:
        print("Serial not connected: Right")