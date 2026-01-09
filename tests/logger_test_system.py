# tests/run_controller_init.py
from os import wait
from pathlib import Path
import time
import sys
import numpy as np
import matplotlib.pyplot as plt  # NEW: for device plots

from PFAS_CTRL.system.pfas_controller import PFASController, PumpBusConfig, PHBusConfig, FluorideBusConfig

# <<< NEW: import logger >>>
from logger import TimelineLogger, plot_timeline
# create global timeline for this run
timeline = TimelineLogger()
# --------------------------------------


# --------------------------------------
# NEW: Device activity recorder
# --------------------------------------
class DeviceActivityRecorder:
    """
    Records when each device (pump, valve, sensor) is active.
    Time axis is in *wall clock seconds* since this script started.
    We later discretize into 1 s bins.
    """
    def __init__(self, dt=1.0):
        self.dt = float(dt)
        self.t0 = time.time()
        # list of (device_name, t_start, t_end, value)
        self.activities = []

    def now(self) -> float:
        """Current time [s] since start of script."""
        return time.time() - self.t0

    def add_interval(self, device: str, t_start: float, t_end: float, value: float = 1.0):
        """Register that `device` is active from t_start to t_end with value."""
        if t_end < t_start:
            t_start, t_end = t_end, t_start
        self.activities.append((device, float(t_start), float(t_end), float(value)))

    def build_vectors(self):
        """
        Build:
            t_axis: 1D array of times [s] with step = dt
            vectors: dict {device_name: 1D array of same length as t_axis}
        Pumps will hold PWM value in [-1,1]; valves/sensors are 0/1.
        """
        if not self.activities:
            return np.array([0.0]), {}

        t_max = max(end for (_, _, end, _) in self.activities)
        # 1-second resolution
        t_axis = np.arange(0.0, np.ceil(t_max) + self.dt, self.dt)
        vectors = {}

        for device, t_start, t_end, value in self.activities:
            if device not in vectors:
                vectors[device] = np.zeros_like(t_axis, dtype=float)

            i0 = int(np.floor(t_start / self.dt))
            i1 = int(np.ceil(t_end   / self.dt))
            i0 = max(i0, 0)
            i1 = min(i1, len(t_axis))

            # Store this interval's value directly (keeps correct PWM, including sign)
            if i0 < i1:
                vectors[device][i0:i1] = value

        return t_axis, vectors


# Create global recorder
device_rec = DeviceActivityRecorder(dt=1.0)
# --------------------------------------


# Helper to extract PWM from info dictionaries
def extract_pwm(info, default_pct: float | None = None) -> float:
    """
    Extract PWM in [0,1] from an info dict returned by controller functions.

    Looks for (in order):
      - 'pump_speed_percent'
      - 'speed_pct'
      - 'speed_percent'

    If none found, falls back to default_pct (if given) else 0.0.
    """
    if isinstance(info, dict):
        if "pump_speed_percent" in info:
            pct = info["pump_speed_percent"]
        elif "speed_pct" in info:
            pct = info["speed_pct"]
        elif "speed_percent" in info:
            pct = info["speed_percent"]
        elif default_pct is not None:
            pct = default_pct
        else:
            return 0.0
    else:
        if default_pct is None:
            return 0.0
        pct = default_pct

    return float(pct) / 100.0


# --- LOAD CONFIG + PARAMS -----

# import system 
try:
    from E_TF_MultipleBatch_Adaptive_c.integrator import RungeKuttaIntegratorCell
except Exception:
    here = Path(__file__).resolve().parent
    model_dir = (here / ".." / "Models_Multiple_Scripts" / "E_TF_MultipleBatch_Adaptive_c").resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from integrator import RungeKuttaIntegratorCell 

try:
    from PFASProject.predictor.helper_functions_ekf import init_EKF
except Exception:
    here = Path(__file__).resolve().parent
    model_dir = (here / ".." / "predictor").resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from helper_functions_ekf import init_EKF

try:
    from PFASProject.predictor.covariance_builder import make_covariances_for_fluoride_only, compute_dynamic_R_from_measurement
except Exception:
    here = Path(__file__).resolve().parent
    pred_dir = (here / ".." / "predictor").resolve()
    if str(pred_dir) not in sys.path:
        sys.path.insert(0, str(pred_dir))
    from covariance_builder import make_covariances_for_fluoride_only, compute_dynamic_R_from_measurement

controller_dir = Path(__file__).resolve().parents[1] / "Controller"
if str(controller_dir) not in sys.path:
    sys.path.insert(0, str(controller_dir))

# now you can import directly
from helper_functions import (
    find_project_root, load_yaml_params, load_yaml_constants,
    load_yaml_covariance, estimate_e, DEFAULT_WEIGHTS,
    vol_from_deltaC_safe, advance_one_control_step,
    build_mpc_adi, mpc_adi, make_normalizers_from_numpy
)
from live_plotter import LiveMPCPlot, predict_horizon_old

# Find parameters
here = Path(__file__).resolve().parent
root = find_project_root(here)
cfg_dir = root / "config"


# load parameters
params, init_vals = load_yaml_params(cfg_dir)
pH = float(init_vals["pH"])
c_cl_0 = float(init_vals["c_cl_0"])
c_so3_0 = float(init_vals["c_so3_0"])
dt_sim = 1.0
k_values = load_yaml_constants(cfg_dir)
k1, k2, k3, k4, k5, k6, k7 = [k_values[f'k{i}'] for i in range(1, 8)]
initial_state = np.array([init_vals["c_pfas_init"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
initial_state = initial_state.reshape((1,1,8)).astype(np.float32) 

#cov_params = load_yaml_covariance(cfg_dir)

# Catalyst concentrations
c_cl = params['c_cl']  # M
c_so3 = params['c_so3']  # M
C_c = [c_so3, c_cl]  # catalyst stock concentrations

# Catalyst limits
cl_max = c_cl*0.1  
so3_max = c_so3*0.1  
u_max   = [so3_max, cl_max]

# Determine sampling time (loop time) Ts
e_max = estimate_e(params, c_so3=so3_max, c_cl=cl_max, pH=pH, c_pfas_init=init_vals["c_pfas_init"], k1=k1)
k_max = max([k1, k2, k3, k4, k5, k6, k7])
Ts = 92.5  # int(1.0 / (k_max * e_max))  # expand later to use func
print(f"Chosen sampling time Ts: {Ts} seconds")

# number of batches
t_r = init_vals["t_r"]
n_batches = init_vals["n_batches"]

# normalizers so costs are O(1)
x0_flat = initial_state.reshape(-1)         # shape (8,)
weights = DEFAULT_WEIGHTS

# Volume parameters
Vi = init_vals["Vi"]*1000 # initial volume [mL]
Vr = init_vals["Vr"]*1000 # reactor volume
Vmax = init_vals["Vmax"]*1000 # maximum volume
V_sens = init_vals["V_sens"]*1000 # volume sampled each step

# integration cell
rk_cell = RungeKuttaIntegratorCell(
        k1, k2, k3, k4, k5, k6, k7,
        params, c_cl_0, c_so3_0, pH, dt_sim,
        initial_state.reshape(1,8), for_prediction=False
    )
rk_cell.build(input_shape=initial_state.shape)

# For plotting / MPC
xk_state = initial_state.copy()              # shape (1,1,8)
x0_flat  = initial_state.reshape(-1)         # shape (8,)
all_states = [x0_flat]
all_inputs = []
measured_F = [0.0]  # initial measurement (dummy)
uk_prev = np.array([0.000,0.000], dtype=float)
x_scale, u_scale = make_normalizers_from_numpy(x0_flat, u_max)

# EKF related
e_init = estimate_e(params, c_so3=uk_prev[0], c_cl=uk_prev[1], pH=pH, c_pfas_init=init_vals["c_pfas_init"], k1=k1)
Q, R, P0 = make_covariances_for_fluoride_only(
    x_scale=x_scale,      # from make_normalizers_from_numpy()
    meas_std=2e-14,       # matches your simulated fluoride noise std
    p0_frac=0.5,          # initial uncertainty = 50% of state magnitude
    rho=0.0               # no inter-state correlation (keep 0 unless needed)
)

# --- STEP 0: Initialize controller -----
ctrl = PFASController(
    pump_cfg=PumpBusConfig(port="/dev/ttyUSB0"),
    ph_cfg=PHBusConfig(port="/dev/ttyUSB1", device_id=2),
    fluoride_cfg=FluorideBusConfig(port="/dev/ttyUSB1", device_id=1),
    logger=timeline,                       # <<< IMPORTANT: hook logger into controller
)

substeps = round(Ts / dt_sim)
print("Substeps per control interval:", substeps)
ctx_adi = build_mpc_adi(
    params=params,
    k_list=[k1, k2, k3, k4, k5, k6, k7],
    pH=pH,
    c_pfas_init=init_vals["c_pfas_init"],
    dt=dt_sim,
    substeps=substeps,
    N=3,
    weights=weights,
    u_max=u_max,
    x0_flat=x0_flat,             
    enable_volume_constraints=True,
    du_max=None,
    rk_cell=rk_cell,
)

ekf = init_EKF(
    x0=np.array(initial_state).reshape(1, 8),
    dt_sim=Ts,
    c_eaq=e_init,
    k=[k1, k2, k3, k4, k5, k6, k7],
    cov_params={"Q": Q, "R": R, "P0": P0},  
    x_scale=x_scale
)
ekf.use_adaptive_R = False
print(f"[ekf-init] id={id(ekf)} use_adaptive_R={ekf.use_adaptive_R} R_floor={ekf.R_floor}")

# --- STEP 1: fill five reservoir tubes ----- 
timeline.start_event("Prime tubes")
prime_tubes_start = device_rec.now()

info = ctrl.initaize()
# Pump ["pfas", "c1", "c2", "buffer", "water"] at 99% speed (fixed)
prime_tubes_end = device_rec.now()
for pump in ["pfas", "c1", "c2", "buffer", "water"]:
    device_rec.add_interval(pump, prime_tubes_start, prime_tubes_end, value=0.99)

timeline.end_event("Prime tubes")

nbatch = 2

# --- STEP 2: create mixture -----
timeline.start_event("Create solution")
create_start = device_rec.now()

info = ctrl.create_mixture(Vi, pfas=1.0, c1=0.0, c2=0.0, speed=99, sequential=False)
create_end = device_rec.now()
val = extract_pwm(info, default_pct=99.0)
device_rec.add_interval("pfas", create_start, create_end, value=val)

timeline.end_event("Create solution")
print(info)

Vs = Vi
xk_flat = initial_state.reshape(-1)  # (8,)
uk_prev = np.array([0.0, 0.0], dtype=float)
xk_sum = np.sum(xk_flat)
eps = 1e-12

cycle_idx = 0

for i in range(nbatch):
    print(i)
    cycle_idx += 1
    timeline.mark_cycle(cycle_idx)

    # --- STEP 3: Run mixture in Reactor -----
    timeline.start_event("Prime reactor")
    prime_reactor_start = device_rec.now()

    ctrl.initialize_reactors()  # fill reactor tubes
    prime_reactor_end = device_rec.now()
    # no info from initialize_reactors, assume 99%
    device_rec.add_interval("pump_mix", prime_reactor_start, prime_reactor_end, value=0.99)

    timeline.end_event("Prime reactor")

    timeline.start_event("Reactor circulation")
    react_start = device_rec.now()

    info = ctrl.supply_reactor(reaction_time_s=Ts, dosage_ml=Vs, cw=True)
    print(info)
    react_end = device_rec.now()
    val = extract_pwm(info, default_pct=99.0)
    device_rec.add_interval("pump_mix", react_start, react_end, value=val)

    timeline.end_event("Reactor circulation")

    # --- STEP 4: Sensor sample -----
    timeline.start_event("prime sensor line")
    prime_sensor_start = device_rec.now()

    # init sensor lines (no info dict, assume 99% during prime)
    ctrl.initialize_sensor()
    prime_sensor_mid = device_rec.now()
    device_rec.add_interval("pump_holding_to_valves", prime_sensor_start, prime_sensor_mid, value=0.99)
    device_rec.add_interval("valve2", prime_sensor_start, prime_sensor_mid, value=1.0)

    # send sensor sample
    info = ctrl.dispatch_sensor(volume_ml=V_sens, speed_pct=99, buffer_pct=0.5)
    print(info)
    prime_sensor_end = device_rec.now()
    val = extract_pwm(info, default_pct=99.0)
    device_rec.add_interval("pump_holding_to_valves", prime_sensor_mid, prime_sensor_end, value=val)
    device_rec.add_interval("buffer", prime_sensor_mid, prime_sensor_end, value=val)
    device_rec.add_interval("valve2", prime_sensor_mid, prime_sensor_end, value=1.0)

    timeline.end_event("prime sensor line")

    # --- STEP 5: Resend rest of volume to stirrer -----
    timeline.start_event("Solution to mixer")
    solution_mixer_start = device_rec.now()

    info = ctrl.dispatch_stirrer_rest(total_ml=Vs, already_sent_ml=V_sens, speed_pct=99)
    print(info)
    solution_mixer_end = device_rec.now()
    val = extract_pwm(info, default_pct=99.0)
    device_rec.add_interval("pump_holding_to_valves", solution_mixer_start, solution_mixer_end, value=val)

    timeline.end_event("Solution to mixer")

    # --- STEP 6: empty sensor line to waste -----
    timeline.start_event("empty sensor lines")
    sample_sensor_start = device_rec.now()

    info = ctrl.dispatch_sensor(volume_ml=5, speed_pct=99, buffer_pct=0.0)
    print(info)
    sample_sensor_end = device_rec.now()
    val = extract_pwm(info, default_pct=99.0)
    device_rec.add_interval("pump_holding_to_valves", sample_sensor_start, sample_sensor_end, value=val)
    device_rec.add_interval("valve2", sample_sensor_start, sample_sensor_end, value=1.0)

    timeline.end_event("empty sensor lines")
    Vs -= V_sens

    # --- STEP 7: Read sensor data and update EKF -----
    timeline.start_event("measurement")
    meas_start = device_rec.now()

    F_M = None
    pH_value = None

    # 1) Fluoride measurement (mol/L)
    if ctrl.fluoride:
        ctrl.fluoride.open()
        F_mgL = ctrl.fluoride.read()
        ctrl.fluoride.close()

        F_M = (float(F_mgL) / 19000.0) / 1000.0  # mg/L -> mol/L
        timeline.log("fluoride_M", F_M)
        meas_f_start = meas_start
        meas_f_end   = device_rec.now()
        device_rec.add_interval("sensor_fluoride", meas_f_start, meas_f_end, value=1.0)

    # 2) pH measurement (unitless)
    if ctrl.ph:
        ctrl.ph.open()
        pH_value = float(ctrl.ph.read())
        ctrl.ph.close()

        timeline.log("pH", pH_value)
        print(f"pH: {pH_value}")
        meas_ph_start = device_rec.now()
        meas_ph_end   = meas_ph_start + 1.0  # approx 1 s
        device_rec.add_interval("sensor_pH", meas_ph_start, meas_ph_end, value=1.0)

    # 3) EKF update + fake PFAS dynamics (only if we have fluoride)
    if F_M is not None:
        # dynamic R from measurement
        R_dynamic = compute_dynamic_R_from_measurement(
            measurement=F_M,
            rel_accuracy=0.02,
            min_std=2e-20,
            units="M",
        )
        ekf.set_R(R_dynamic)

        # hydrated electrons for applied uk
        ekf.set_c_eaq(estimate_e(
            params,
            c_so3=uk_prev[0],
            c_cl=uk_prev[1],
            pH=pH,
            c_pfas_init=xk_flat[0],
            k1=k1,
        ))

        # EKF predict-update
        ekf.predict(xk_flat)
        xk_state = ekf.update(max(F_M, 0.0))
        xk_flat = np.asarray(xk_state).reshape(-1).astype(float)
        xk_sum = np.sum(xk_flat)

        # TEMP: simulate degradation since there is no real PFAS yet
        if i == 1:
            xk_flat = initial_state.reshape(-1)
            xk_sum = np.sum(xk_flat)
        else:
            xk_flat = np.zeros_like(xk_flat)
            xk_sum = 0.0

    meas_end = device_rec.now()
    timeline.end_event("measurement")

    # --- STEP 8: Clean sensors with water -----
    timeline.start_event("Clean sensor")
    clean_start = device_rec.now()

    info = ctrl.flush_sensor_water(volume_ml=2, speed_pct=99)
    print(info)
    clean_end = device_rec.now()
    val = extract_pwm(info, default_pct=99.0)
    device_rec.add_interval("water", clean_start, clean_end, value=val)

    timeline.end_event("Clean sensor")

    # --- STEP 9: Add catalysts (MPC) -----
    timeline.start_event("Actuation")
    actuation_start = device_rec.now()

    uk, Uplan, Jstar = mpc_adi(
        xk_flat=xk_flat, uk_prev=uk_prev, ctx=ctx_adi,
        Vs0=Vs/1000.0,          # L
        V_sens=V_sens/1000.0,   # L
        V_max=Vmax/1000.0,      # L
        C_c=C_c, warm_start=None
    )
    plan = ctrl.mapMPC2pump(
        u_prev=uk_prev, u_k=uk,
        Vs_ml=Vs,
        Cc_so3=C_c[0], Cc_cl=C_c[1],
        pump_so3="c1", pump_cl="c2",
        speed_pct=99.0,  # full speed (upper bound)
    )
    actuation_end = device_rec.now()
    val = extract_pwm(plan, default_pct=99.0)

    # --- SPECIAL CASE ---
    # If this is the *second* cycle (i == 2), force catalyst PWM = 0
    if i == 1:
        val_c1 = 0.0
        val_c2 = 0.0
    else:
        val_c1 = val
        val_c2 = val

    device_rec.add_interval("c1", actuation_start, actuation_end, value=val_c1)
    device_rec.add_interval("c2", actuation_start, actuation_end, value=val_c2)

    timeline.end_event("Actuation")
    
    # update properties
    Vs = plan["Vs_ml_after"]  # update volume
    print(f"Updated volume after catalyst addition: {Vs} mL")
    uk_prev = uk.copy()


# --- STEP 10: Final flush -----
timeline.start_event("Empty tubes")
empty_start = device_rec.now()

info = ctrl.empty_tubes(volume_ml=Vs, speed_pct=99)
print(info)
empty_end = device_rec.now()

# Use actual PWM magnitude if available, otherwise 99%
pwm_val = extract_pwm(info, default_pct=99.0)
back_val = -abs(pwm_val)

# Pumps that should be logged with NEGATIVE PWM during emptying
neg_pumps = ["pfas", "c1", "c2", "buffer", "water"]
for pump in neg_pumps:
    device_rec.add_interval(pump, empty_start, empty_end, value=back_val)

# Other pumps still logged as positive
for pump in ["pump_mix", "pump_holding_to_valves"]:
    device_rec.add_interval(pump, empty_start, empty_end, value=pwm_val)

device_rec.add_interval("valve1", empty_start, empty_end, value=1.0)
device_rec.add_interval("valve2", empty_start, empty_end, value=1.0)

timeline.end_event("Empty tubes")


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

# --- FINAL: make timing plot -----
plot_timeline(
    timeline,
    title="PFAS System Timing (Controller Test Run)",
    filename="system_timing_run.png",
    show=False,
)
print("Saved system_timing_run.png")


# --- NEW FINAL: Build per-device vectors (1 s resolution), save, and plot ---
t_axis, device_vectors = device_rec.build_vectors()

if device_vectors:
    # Save all vectors in a single NPZ file
    save_dict = {"time_s": t_axis}
    save_dict.update(device_vectors)
    np.savez("system_device_activity_vectors.npz", **save_dict)
    print("Saved device activity vectors to system_device_activity_vectors.npz")

    # Quick per-device step plot (debug)
    dev_names = sorted(device_vectors.keys())
    n_dev = len(dev_names)

    fig, axes = plt.subplots(n_dev, 1, sharex=True, figsize=(12, 1.5 * n_dev), constrained_layout=True)
    if n_dev == 1:
        axes = [axes]

    for ax, name in zip(axes, dev_names):
        vec = device_vectors[name]
        ax.step(t_axis, vec, where="post")
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Pump / Valve / Sensor activity (1 s resolution, value = PWM or on/off)")
    fig.savefig("system_device_timeline.png", dpi=300)
    plt.close(fig)
    print("Saved system_device_timeline.png")
else:
    print("No device activity recorded – no device timeline produced.")
