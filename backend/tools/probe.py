"""Probe serial connectivity without sending a movement command."""

from backend.hardware.serial_connection import find_esp32


def main():
    print(find_esp32() or "No responding Camman device found")


if __name__ == "__main__":
    main()
