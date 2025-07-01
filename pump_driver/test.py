import serial
import time
from pump_class import WX10Pump

port = "/dev/ttyUSB0"   # Adjust as needed
baudrate = 9600

serial_port = serial.Serial(
            port=port, 
            baudrate=baudrate,
            bytesize=8, 
            parity=serial.PARITY_EVEN,
            stopbits=1, 
            timeout=1
        )
serial_port.reset_output_buffer()  # Clear any existing data in the buffer
# Instance of class WX10Pump
pump = WX10Pump(port=serial_port, address=2, baudrate=baudrate)
#pump.close()  # Close any existing connection


pump.set_speed(50.0)     # Start at 50 RPM CW
pump.set_address(address=3)  # Change address to 1
pump.set_speed(50.0)     # Start at 50 RPM CW
time.sleep(5) 

pump.set_address(address=2)  # Change address to 2
pump.stop()              # Stop pump
pump.set_address(address=3)  # Change address to 2
pump.stop()              # Stop pump
pump.close()




