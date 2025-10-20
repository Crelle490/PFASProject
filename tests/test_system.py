# tests/run_controller_init.py
from pathlib import Path
import time
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PFAS_CTRL.system.pfas_controller import PFASController, PumpBusConfig, PHBusConfig, FluorideBusConfig

ctrl = PFASController(
    pump_cfg=PumpBusConfig(port="/dev/ttyUSB1"),
    ph_cfg=PHBusConfig(port="/dev/ttyUSB0", device_id=2),
    fluoride_cfg=FluorideBusConfig(port="/dev/ttyUSB0", device_id=1),
)

# --- STEP 1: fill five resvior tubes ----- 
batch_ml = 25
"""
sensor_batch = 5
nbatch = 1
info = ctrl.initaize()


# --- STEP 2: create mixture -----
info = ctrl.create_mixture(batch_ml, pfas=0.5, c1=0.25, c2=0.25, speed=99, sequential=False)
print(info)


for i in range(nbatch):
    # --- STEP 3: Run mixture in Reactor -----
    ctrl.initialize_reactors() # fill pump tubes
    info = ctrl.supply_reactor(reaction_time_s=50, dosage_ml=batch_ml, cw=True)
    print(info)
    ctrl.initialize_reactors() # empty pump tubes

    # --- STEP 4: Sensor sample -----
    ctrl.initialize_sensor() # prime sensor line
    info = ctrl.dispatch_sensor(volume_ml=sensor_batch, speed_pct=99, buffer_pct=0.2)
    print(info)

    # --- STEP 5: Resend rest of volume to stirrer -----
    infor = ctrl.dispatch_stirrer_rest(total_ml=batch_ml, already_sent_ml=sensor_batch, speed_pct=99)
    print(infor)


    # --- STEP 6: Pump tubes to sensor clean -----
    info = ctrl.dispatch_sensor(volume_ml=15, speed_pct=99, buffer_pct=0)


    # --- STEP 7: Clean sensors -----
    ctrl.flush_sensor_water(volume_ml=10, speed_pct=99)


    # --- STEP 8: Add catalysts -----
    # control logic: u = f() = add_batch
    add_batch = 5
    info = ctrl.create_mixture(add_batch, pfas=0, c1=0.5, c2=0.5, speed=99, sequential=False)
    
    batch_ml = batch_ml - sensor_batch + add_batch # update batch size

# --- STEP 0: Final flush -----
ctrl.exit_fluid(volume_ml=batch_ml, speed_pct=99.0)
"""
ctrl.empty_tubes(volume_ml=batch_ml, speed_pct=50)


# Sensors (manual open/close)
if ctrl.ph:
    ctrl.ph.open()
    print("pH:", ctrl.ph.read())
    ctrl.ph.close()

if ctrl.fluoride:
    ctrl.fluoride.open()
    print("F mg/L:", ctrl.fluoride.read())
    ctrl.fluoride.close()

ctrl.close()
