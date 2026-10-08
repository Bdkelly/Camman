"""Camman/1: newline-delimited, normalized pan velocity over any byte transport."""

from backend.control import normalized_pan

PROTOCOL_GREETING = "CAMMAN/1 VELOCITY"


def velocity_command(pan, *, invert=False):
    value = normalized_pan(pan) * (-1 if invert else 1)
    return "Stop" if value == 0 else f"V:{value:.4f}"
