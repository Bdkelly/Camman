import serial
import serial.tools.list_ports
import time

def send_test(conn, command):
    conn.write((command + '\n').encode('utf-8'))
    print(f"-> Sent: {command}")
    time.sleep(0.1)
    response = conn.readline().decode('utf-8').strip()
    return response

def valid_serial(port):
    brate, scommand, rcommand, timeout = config()
    ser = serial.Serial(port.device, brate, timeout=timeout)
    ser.reset_input_buffer()
    response = send_test(ser, scommand,rcommand)
    try:
        if response == rcommand:
            print(f"Connection Valid {port}")
            return ser
        else:
            print(f"Connection NOT Valid {port}")
            ser.close()
    except:
        print(f"Some Error Has occered at {port}")
        ser.close()
    return "X"

def config():
    brate = 115200
    scommand = "Right"
    rcommand = "OK: Moving Right"
    timeout = 1
    return brate, scommand, rcommand, timeout


if __name__ == "__main__":
    valid_serial()
