"""Probe the firmware's Stop response without requesting a motor movement."""

import serial


def valid_serial(port):
    try:
        with serial.Serial(port.device, 115200, timeout=1, write_timeout=1) as conn:
            conn.reset_input_buffer()
            conn.write(b"Stop\n")
            return conn.readline().decode("utf-8", errors="replace").strip() == "Stopping"
    except (serial.SerialException, OSError):
        return False
