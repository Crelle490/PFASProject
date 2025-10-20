import serial
import time
from pump_class import WX10Pump
import gpiod

port = "/dev/ttyUSB1"   # Adjust as needed
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
pump = WX10Pump(port=serial_port, address=1)
#pump.close()  # Close any existing connection
pump.set_speed(50.0)
time.sleep(0.5)

for i in range(2,12):
    pump.set_address(address=i)
    pump.set_speed(50.0)    # Start at 100 RPM CW
    time.sleep(1)

# relay control
time.sleep(5)
for i in range(1,12):
    pump.set_address(address=i)
    pump.stop()              # Stop pump
pump.close()

pump.control_io("valve1", "on")                 # tænd ventil 1 (GPIO23)
time.sleep(1)
pump.control_io("valve2", "on")                # sluk ventil 2 (GPIO24)
time.sleep(1)
pump.control_io("fans", "on")                # tænd fans (GPIO25)
time.sleep(10)
pump.control_io("valve1", "off")                # sluk ventil 1 (GPIO23)
pump.control_io("valve2", "off")                 # tænd ventil 2 (
pump.control_io("fans", "off")                 # sluk fans (GPIO25)
#pump.control_io("fans", "blink", period_s=2)    # blink fans (GPIO25) hvert 2. sekund

pump.print_flow("flow1", seconds=8)
pump.print_flow("flow2", seconds=5)






#pump.set_speed(50.0)     # Start at 50 RPM CW
#pump.set_address(address=3)  # Change address to 1

#time.sleep(5) 

#pump.set_address(address=2)  # Change address to 2
#pump.stop()              # Stop pump
#pump.set_address(address=3)  # Change address to 2
#pump.stop()              # Stop pump
#pump.close()




