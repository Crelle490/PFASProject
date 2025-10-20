import serial, time
from pump_class import WX10Pump

ser = serial.Serial("/dev/ttyUSB1", 9600, bytesize=8, parity=serial.PARITY_EVEN, stopbits=1, timeout=0.5)

pump = WX10Pump(port=ser, address=31)  # broadcast for set-only
#pump.set_speed(50.0, run=True, broadcast=True)  # all pumps run, no reply
#time.sleep(2.0)
#pump.stop(address=31)

# Unicast to a single unit (e.g., 4,5,6,7)
pump.control_io("fans", "on") # tænd fans (GPIO25)
time.sleep(1)
for a in (1,2,3,4,5,6,7):
    pump.set_address(a)
    pump.set_speed(40.0, run=True, cw=True)
    print(a, pump.get_state())

time.sleep(1.0)
pump.control_io("fans", "off") # tænd fans (GPIO25)
pump.close()




