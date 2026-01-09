

#!/usr/bin/env python3
# tests/run_pump_min.py

# -- Import Dependencies --
import sys
from pathlib import Path
import serial

# make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PFAS_CTRL.drivers.pump_wx10 import WX10Pump
from PFAS_CTRL.drivers.flow_slf3c import SLF3C
from PFAS_CTRL.drivers.ph_sensor import SerialConfig, PHAnalyzer
from PFAS_CTRL.drivers.flouride_sensor import SerialConfig, FluorideAnalyzer
from PFAS_CTRL.drivers.gpio_control import GPIOCtrl
from PFAS_CTRL.drivers.orion_sensor import SerialConfig, OrionVersaStarPro
import time


# -- Pump Test -- doesnt work with single address!!!
ser = serial.Serial("/dev/ttyUSB0", 9600, bytesize=8,
                    parity=serial.PARITY_EVEN, stopbits=1, timeout=0.5)

gpio = GPIOCtrl(active_low=False).open()
#gpio.on("valve3")
#time.sleep(2)
#gpio.off("valve3")

#orion = OrionVersaStarPro(SerialConfig(port="/dev/ttyACM0", baudrate=115200,
#                                 parity="EVEN", stopbits=1, timeout=1.0))
#orion.open()
#try:
#    print("One-shot ORP:", orion.read())
#    print("3 seconds @ 5 Hz:", orion.read(seconds=0.5, hz=5))
#finally:
#    orion.close()

#gpio.on("fans")
#time.sleep(2)
#gpio.off("fans")
#gpio.off("valve1")
#gpio.on("valve2")
gpio.on("valve3")
pump   = WX10Pump(port=ser, address=31)
pump.set_speed(50, cw=False)  # ensure stopped
time.sleep(30)
pump.stop()


# Results; 2: 7.20, 3: 7.48, 7: 7.53, 5: 8.3, 4: 8.68

"""
ADDRS  = [1,2,3,4,5,6,7]  # your pump addresses
STEP_DELAY = 0.02         # seconds per rpm step (bus-friendly)
PHASE      = 0.05         # seconds stagger between pumps
HOLD_S     = 2.064

t0 = time.time()
last = {a: -1 for a in ADDRS}
while True:
    t = time.time() - t0
    done = True
    for i, addr in enumerate(ADDRS):
        k = int((t - i*PHASE) / STEP_DELAY)  # step index for this pump
        if k < 0:   k = 0
        if k > 99:  k = 99
        if k != last[addr]:
            pump.set_speed(k, run=True, cw=True, address=addr)
            last[addr] = k
        if k < 99:
            done = False
    if done:
        break
    time.sleep(0.005)
time.sleep(HOLD_S)
pump.set_speed(0.0, run=False, broadcast=True)  # stop all

# -- Flow Sensor Test --
fs = SLF3C()
print(fs.read("flow1"))                 # one sample
print(fs.read("flow2", seconds=0.5, hz=5))  # list of ~15 samples


# -- PH sensor --
cfg = SerialConfig(port="/dev/ttyUSB0", baudrate=9600, parity="N", stopbits=1, timeout=1.0)
ph = PHAnalyzer(cfg, device_id=2)
ph.open()
try:
    print("One-shot pH:", ph.read())
    print("3 seconds @ 5 Hz:", ph.read(seconds=0.5, hz=5))
finally:
    ph.close()



# -- Flouride sensor --
cfg = SerialConfig(port="/dev/ttyUSB0", baudrate=9600, parity="E", stopbits=1, timeout=1.5)
with FluorideAnalyzer(SerialConfig(port="/dev/ttyUSB0"), device_id=1) as fa:
    print("One-shot mg/L:", fa.read())
    print("3s @ 5 Hz:", fa.read(seconds=0.5, hz=5))



# -- GPIO Control --
io = GPIOCtrl(active_low=False).open()
io.on("valve1")
io.on("valve2")          # valve1 stays on
io.on("fans")
time.sleep(5)
io.set_many({"valve1": False, "valve2": False, "fans": False})
io.blink("valve2", period_s=0.5, cycles=2)
io.close()
