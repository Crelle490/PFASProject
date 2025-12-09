# tests/run_controller_init.py
from pathlib import Path
import time
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from PFAS_CTRL.system.pfas_controller import PFASController, PumpBusConfig, PHBusConfig, FluorideBusConfig



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
c_cl_0 = float(init_vals["c_cl"])
c_so3_0 = float(init_vals["c_so3"])
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
Ts = 75#int(1.0 / (k_max * e_max))  # expand later to use func
print(f"Chosen sampling time Ts: {Ts} seconds")

# number of batches
t_r = init_vals["t_r"]
n_batches = init_vals["n_batches"]

# normalizers so costs are O(1)
x0_flat = initial_state.reshape(-1)         # shape (8,)
weights = DEFAULT_WEIGHTS

# Volume parameters
Vi = init_vals["Vi"]*1000 # initial volume
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

# For plotting
xk_state = initial_state.copy()              # shape (1,1,8)
x0_flat  = initial_state.reshape(-1)         # shape (8,)
all_states = [x0_flat]
all_inputs = []
measured_F = [0.0]  
uk_prev = np.array([0.000,0.000], dtype=float)
x_scale, u_scale = make_normalizers_from_numpy(x0_flat, u_max)

# EKF related
e_init = estimate_e(params, c_so3=uk_prev[0], c_cl=uk_prev[1], pH=pH, c_pfas_init=init_vals["c_pfas_init"], k1=k1)
Q, R, P0 = make_covariances_for_fluoride_only(
    x_scale=x_scale,      # from make_normalizers_from_numpy()
    meas_std=2e-14,        # matches your simulated fluoride noise std
    p0_frac=0.5,          # initial uncertainty = 50% of state magnitude
    rho=0.0               # no inter-state correlation (keep 0 unless needed)
    )


# --- STEP 0: Initialize controller -----
ctrl = PFASController(
    pump_cfg=PumpBusConfig(port="/dev/ttyUSB1"),
    ph_cfg=PHBusConfig(port="/dev/ttyUSB0", device_id=2),
    fluoride_cfg=FluorideBusConfig(port="/dev/ttyUSB0", device_id=1),
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





# --- STEP 1: fill five resvior tubes ----- 
info = ctrl.initaize()
nbatch = 5

# --- STEP 2: create mixture -----
info = ctrl.create_mixture(Vi, pfas=0.5, c1=0.25, c2=0.25, speed=99, sequential=False)
print(info)
Vs = Vi
xk_flat = initial_state.reshape(-1)  # (8,)
uk_prev = np.array([0.0, 0.0], dtype=float)
xk_sum = np.sum(xk_flat)
eps = 1e-12

#for i in range(nbatch):
while xk_sum > eps:
    # --- STEP 3: Run mixture in Reactor -----
    ctrl.initialize_reactors() # fill pump tubes
    info = ctrl.supply_reactor(reaction_time_s=50, dosage_ml=Vs, cw=True)
    print(info)
    ctrl.initialize_reactors() # empty pump tubes

    # --- STEP 4: Sensor sample -----
    ctrl.initialize_sensor() # prime sensor line
    info = ctrl.dispatch_sensor(volume_ml=V_sens, speed_pct=99, buffer_pct=0.2)
    print(info)

    # --- STEP 5: Resend rest of volume to stirrer -----
    info = ctrl.dispatch_stirrer_rest(total_ml=Vs, already_sent_ml=V_sens, speed_pct=99)
    print(info)


    # --- STEP 6: Pump tubes to sensor clean -----
    info = ctrl.dispatch_sensor(volume_ml=15, speed_pct=99, buffer_pct=0)
    Vs -= V_sens

    # --- STEP 7: Read sensor data and update EKF -----
    if ctrl.fluoride:
        ctrl.fluoride.open()
        F_mgL = ctrl.fluoride.read()
        ctrl.fluoride.close()
        F_M = (float(F_mgL) / 19000.0) / 1000.0

        R_dynamic = compute_dynamic_R_from_measurement(
            measurement=F_M, rel_accuracy=0.02, min_std=2e-20, units="M"
        )
        ekf.set_R(R_dynamic)

        # set c_eaq for applied uk (now defined)
        ekf.set_c_eaq(estimate_e(params, c_so3=uk_prev[0], c_cl=uk_prev[1], pH=pH,
                                c_pfas_init=xk_flat[0], k1=k1))

        # Predict with last state, then update with **M**
        ekf.predict(xk_flat)
        xk_state = ekf.update(max(F_M, 0.0))
        xk_flat = np.asarray(xk_state).reshape(-1).astype(float)
        xk_sum = np.sum(xk_flat)

    # --- STEP 8: Clean sensors -----
    ctrl.flush_sensor_water(volume_ml=10, speed_pct=99)


    # --- STEP 9: Add catalysts -----
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
        speed_pct=99.0,  # full speed
    )
    
    # update properties
    Vs = plan["Vs_ml_after"]  # update volume
    uk_prev = uk.copy()


# --- STEP 0: Final flush -----
ctrl.exit_fluid(volume_ml=Vs, speed_pct=99.0)

ctrl.empty_tubes(volume_ml=Vs, speed_pct=50)


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
