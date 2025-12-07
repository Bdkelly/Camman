import serial
import serial.tools.list_ports
import time
from guiapp.utils.ser_val import valid_serial

_command_signal_ref = None

def set_command_signal(signal):
    global _command_signal_ref
    _command_signal_ref = signal

def scan_port(port):
    result = valid_serial(port)
    if result == "X":
        return False
    else:
        return True

def send_agent_command(ser, pan):
    if ser:
        command = f"P:{pan:.2f}"
        if _command_signal_ref: 
            _command_signal_ref.emit(command.strip())
        ser.write(command.encode('utf-8'))
        print(f"Sent command: {command.strip()}")
    else:
        print("Serial not connected: Agent Command")   

def find_esp32():
    ports = serial.tools.list_ports.comports()
    print(f"Scanning {len(ports)} serial ports...")
    try:
        for port in ports:
            print(f"Checking port: {port.device} - {port.description}")
            if(scan_port(port)):
                return "Serial Connection Valid: {port.name}"
        return "No Valid Connection"
    except:    
        return "No Valid Connection"

def move_left(ser):
    if ser:
        if _command_signal_ref: _command_signal_ref.emit("Left") # Use internal reference
        ser.write(b"Left\n") 
        print("Sent command: Left")
    else:
        print("Serial not connected: Left")

def move_right(ser):
    if ser:
        if _command_signal_ref: _command_signal_ref.emit("Right") # Use internal reference
        ser.write(b"Right\n") 
        print("Sent command: Right")
    else:
        print("Serial not connected: Right")