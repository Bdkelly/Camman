import serial
import serial.tools.list_ports
import time
import random

def send_test(conn, command):
    print(f"Sending: {command.strip()}")
    conn.write(command.encode('utf-8'))

def testport(port):
    value = random.randint(1, 100) / 100
    print(f"Testing port: {port.device} with value {value}")
    
    brate, timeout = config()
    
    try:
        ser = serial.Serial(port.device, brate, timeout=timeout)
        
        print("Waiting for ESP32 to boot...")
        time.sleep(2) 
        
        cmd = f"P:{value}\n" 
        
        ser.reset_input_buffer()
        send_test(ser, cmd)
        
        time.sleep(0.1)
        if ser.in_waiting:
            response = ser.readline().decode('utf-8').strip()
            print(f"ESP32 Replied: {response}")
            
        ser.close()
        
    except serial.SerialException as e:
        print(f"Could not open {port.device}: {e}")

def config():
    brate = 115200
    timeout = 1
    return brate, timeout

if __name__ == "__main__":
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found!")
    else:
        for port in ports:
            if "COM" in port.device or "USB" in port.device or "tty" in port.device:
                testport(port)