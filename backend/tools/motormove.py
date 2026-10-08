"""Send a bounded manual velocity pulse to a CAMMAN/1 controller."""

import argparse
import math
import time

from backend.hardware.protocol import velocity_command
from backend.hardware.serial_connection import (
    open_connection,
    send_agent_command,
    verify_velocity_controller,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--pan", type=float, help="Normalized velocity in [-1, 1]")
    command.add_argument("--command", choices=["Left", "Right", "Stop"])
    parser.add_argument("--duration", type=float, default=0.2, help="Pulse seconds, maximum 0.5")
    parser.add_argument("--invert-pan", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.duration <= 0.5:
        parser.error("Duration must be 0..0.5 seconds")
    action = (
        args.pan
        if args.pan is not None
        else {"Left": -0.25, "Right": 0.25, "Stop": 0}[args.command]
    )
    if not math.isfinite(action) or not -1 <= action <= 1:
        parser.error("Pan must be a finite value in [-1, 1]")
    connection = None
    try:
        connection = open_connection(args.port)
        if connection is None:
            raise RuntimeError("No controller found")
        verify_velocity_controller(connection)
        send_agent_command(connection, velocity_command(action, invert=args.invert_pan), print)
        if action:
            time.sleep(args.duration)
    finally:
        if connection is not None:
            try:
                send_agent_command(connection, "Stop", print)
            finally:
                connection.close()


if __name__ == "__main__":
    main()
