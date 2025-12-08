import serial
import time
import sys

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
            self.ser.write(full_command.encode('utf-8'))
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

# --- usage example ---
if __name__ == "__main__":
    # REPLACE 'COM3' with your actual port (e.g., '/dev/ttyUSB0' on Linux/Mac)
    PORT = 'COM3' 
    
    motor = MotorController(PORT)

    try:
        # 1. Move using the proportional logic (P command)
        # This will trigger: steps = 0.5 * 200.0 * 1 = 100 steps
        motor.move_pan(0.5) 
        time.sleep(1) # Let motor move

        # 2. Move opposite direction
        motor.move_pan(-0.5)
        time.sleep(1)

        # 3. Use text commands
        motor.move_right()
        time.sleep(0.5)
        
        motor.stop()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        motor.stop()
        motor.close()