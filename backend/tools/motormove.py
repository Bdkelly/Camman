import sys
import time

import serial


class MotorController:
    def __init__(self, port, baud_rate=115200):
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=1)
            # Allow time for the ESP32 to reset after serial connection is established
            print(f"Connecting to {port}...")
            time.sleep(2)
            print("Connected.")
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            sys.exit(1)

    def send_command(self, command_str):
        """
        Sends a command string to the ESP32.
        Appends the newline character as required by Serial.readStringUntil('\n').
        """
        if self.ser.is_open:
            # Ensure command ends with newline
            full_command = f"{command_str}\n"
            self.ser.write(full_command.encode("utf-8"))
            # Optional: Read response if you want to verify (e.g., "Stopping")
            # response = self.ser.readline().decode().strip()
            # if response: print(f"ESP32: {response}")
        else:
            print("Serial port not open.")

    def move_pan(self, value):
        """
        Sends the 'P:' command.
        NOTE: Your firmware requires a comma to parse the float!
        Structure: P:<value>,
        """
        # We append ',0' to ensure the comma exists for the firmware parser
        cmd = f"P:{value},0"
        print(f"Sending Move: {cmd}")
        self.send_command(cmd)

    def move_right(self):
        print("Sending: Right")
        self.send_command("Right")

    def move_left(self):
        print("Sending: Left")
        self.send_command("Left")

    def stop(self):
        print("Sending: Stop")
        self.send_command("Stop")

    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Send one ESP32 motor command")
    parser.add_argument("--port", required=True)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--pan", type=float)
    command.add_argument("--command", choices=["Left", "Right", "Stop"])
    args = parser.parse_args(argv)
    motor = MotorController(args.port)
    try:
        if args.pan is not None:
            motor.move_pan(args.pan)
        else:
            motor.send_command(args.command)
    finally:
        motor.close()


if __name__ == "__main__":
    main()
