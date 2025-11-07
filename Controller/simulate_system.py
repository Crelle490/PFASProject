

import numpy as np
import yaml
import tensorflow as tf
import sys
from pathlib import Path
from helper_functions import find_project_root, load_yaml_params, load_yaml_constants, load_yaml_covariance, estimate_e, DEFAULT_WEIGHTS, vol_from_deltaC_safe
from helper_functions import advance_one_control_step
import matplotlib.pyplot as plt
from helper_functions import build_mpc_adi, mpc_adi, make_normalizers_from_numpy
import casadi as ca
from live_plotter import LiveMPCPlot, predict_horizon_old

# ---- 1. PARAMETERS AND INITIAL CONDITIONS ----




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
Ts = 75#int(1.0 / (k_max * e_max))  # expand later to use func
print(f"Chosen sampling time Ts: {Ts} seconds")

# number of batches
t_r = init_vals["t_r"]
n_batches = init_vals["n_batches"]

# normalizers so costs are O(1)
x0_flat = initial_state.reshape(-1)         # shape (8,)
weights = DEFAULT_WEIGHTS

# Volume parameters
Vi = init_vals["Vi"] # initial volume
Vr = init_vals["Vr"] # reactor volume
Vmax = init_vals["Vmax"] # maximum volume
V_sens = init_vals["V_sens"] # volume sampled each step

# integration cell
rk_cell = RungeKuttaIntegratorCell(
        k1, k2, k3, k4, k5, k6, k7,
        params, c_cl_0, c_so3_0, pH, dt_sim,
        initial_state.reshape(1,8), for_prediction=False
    )
rk_cell.build(input_shape=initial_state.shape)

# EKF 


# Build CasADi MPC context once (outside simulate loop)
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



# ---- 2. MPC CONTROLLER ----

# ---- 3. SIMULATE WHOLE PROCESS ----
def simulate(with_catalyst=True, steps=100,Vi=0):
    

    # timing & limits
    substeps = int(round(Ts / dt_sim))
    du_max   = np.array([0.2 * so3_max, 0.2* cl_max], dtype=float)

    # state for the integrator (tensor-shaped)
    xk_state = initial_state.copy()              # shape (1,1,8)
    x0_flat  = initial_state.reshape(-1)         # shape (8,)

    all_states = [x0_flat]
    all_inputs = []
    measured_F = [0.0]  # initial measurement (dummy)
    uk_prev = np.array([0.000,0.000], dtype=float)
    x_scale, u_scale = make_normalizers_from_numpy(x0_flat, u_max)
    # Initialize EKF c_eaq based on initial inputs
    e_init = estimate_e(params, c_so3=uk_prev[0], c_cl=uk_prev[1], pH=pH, c_pfas_init=init_vals["c_pfas_init"], k1=k1)
   
    # --- Build well-scaled EKF covariances ---
    Q, R, P0 = make_covariances_for_fluoride_only(
        x_scale=x_scale,      # from make_normalizers_from_numpy()
        meas_std=2e-14,        # matches your simulated fluoride noise std
        p0_frac=0.5,          # initial uncertainty = 50% of state magnitude
        rho=0.0               # no inter-state correlation (keep 0 unless needed)
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


    
    t_max   = steps * Ts
    z0_init = float(np.sum(initial_state.reshape(-1)[:7]))
    live = LiveMPCPlot(Ts=Ts, t_max=t_max, z0=z0_init, u_max=u_max, x0=x0_flat, make_state_grid=True)



    try:
        for step in range(steps):
            print(f"Step {step+1}/{steps} - Current state: {all_states[-1]}")

            current_state = all_states[-1]  # shape (1,1,8)
            

            if with_catalyst:
                # --- solve MPC ---
                Z_est = float(np.sum(current_state[:7]))   # ΣPFAS from EKF/plant state
                z_start = 5e-10    # start dosing above this
                z_stop  = 2e-10    # stop dosing below this for H steps
                H = 2              # hysteresis steps

                # no input if PFAS is close to zero
                if step >= H and all(np.sum(all_states[-i][:7]) < z_stop for i in range(1, H+1)):
                    uk = uk_prev # no change in catalyst
                    Uplan = uk_prev[np.newaxis, :].repeat(ctx_adi["N"], axis=0)
                    Jstar = 0.0
                else:
                    uk, Uplan, Jstar = mpc_adi(
                        xk_flat=current_state, uk_prev=uk_prev, ctx=ctx_adi,
                        Vs0=Vi, V_sens=V_sens, V_max=Vmax, C_c=C_c, warm_start=None
                    )
                # --- live prediction/plot BEFORE applying uk ---
                if live is not None:
                    t_pred_rel, Z_pred, t_u_pred_rel, X_pred = predict_horizon_old(
                        ctx_adi, rk_cell, current_state.reshape(1,1,8), Uplan, substeps, Ts
                    )

                    X_hist_arr = np.vstack(all_states)  # (k+1, 8)

                    # Build safe arrays for inputs
                    if len(all_inputs):
                        t_u_hist_arr = np.arange(len(all_inputs)) * Ts
                        U_hist_arr   = np.asarray(all_inputs, dtype=float)
                    else:
                        t_u_hist_arr = np.array([], dtype=float)
                        U_hist_arr   = np.zeros((0, 2), dtype=float)

                    live.update(
                        t_hist=np.arange(len(all_states)) * Ts,
                        Z_hist=np.array([np.sum(x[:7]) for x in all_states], dtype=float),
                        t_u_hist=t_u_hist_arr,
                        U_hist=U_hist_arr,
                        t0_abs=step * Ts,
                        t_pred_rel=t_pred_rel,
                        Z_pred=Z_pred,
                        t_u_pred_rel=t_u_pred_rel,
                        U_plan=np.asarray(Uplan, dtype=float),

                        F_meas_t=np.arange(len(measured_F)) * Ts,
                        F_meas=np.asarray(measured_F, dtype=float),

                        # (optional) full-state panels
                        X_hist=X_hist_arr,
                        X_pred=X_pred,
                    )


                deltaC = uk-uk_prev
                dV = 0.0
                for i in range(2):
                    dV += vol_from_deltaC_safe(deltaC[i], C_c[i], Vi, eps=1e-12)
                print("Change in volume:", dV)
                Vi = float(Vi + dV-V_sens)
                print("New volume:", Vi)

                    
            else:
                # no catalyst case
                Uplan = np.array([[0.0, 0.0]] * ctx_adi["N"], dtype=float)
                uk = np.array([0.0, 0.0], dtype=float)
            

            all_inputs.append(uk.copy())
            uk_prev = uk

            # --- advance plant by one control interval Ts (multiple Δt) ---
            xk_state_simulated = advance_one_control_step(rk_cell, current_state.reshape(1,1,8), uk, int(substeps))
            
            # Reshape to (8,) for EKF prediction
            xk_state_simulated = xk_state_simulated[0]
            xk_state_simulated = np.reshape(xk_state_simulated.numpy(), (-1))  # shape (8,)
            
            # Update R
            R_dynamic = compute_dynamic_R_from_measurement(
                measurement=xk_state_simulated[-1],
                rel_accuracy=0.02,
                min_std=2e-20,
                units="M"
            )
            ekf.set_R(R_dynamic)
            Q_ff = float(ctx_adi["Q"][-1, -1]) if "Q" in ctx_adi else float(Q[-1, -1])
            R_ff = float(R_dynamic[0, 0])
            print(f"R update → {R_ff:.3e}  |  Q_ff/R = {Q_ff/R_ff:.3f}")

            # Set correct c_eaq based on applied uk
            #ekf.set_c_eaq(estimate_e(params, c_so3=uk[0], c_cl=uk[1], pH=pH,c_pfas_init=init_vals["c_pfas_init"], k1=k1))
            ekf.set_c_eaq(estimate_e(params, c_so3=uk[0], c_cl=uk[1], pH=pH,c_pfas_init=current_state[0], k1=k1))
            
            # Insert simulated state into EKF predict step
            ekf.predict(xk_state_simulated)

            # Return state
            noise = np.random.normal(0, 2e-7)
            simulated_flouride = xk_state_simulated[7] + noise  # fluoride measurement with noise
            measured_F.append(simulated_flouride)
            xk_state = ekf.update(np.maximum(simulated_flouride, 0.0))  # fluoride measurement
            #xk_state = xk_state_simulated
            #xk_state = xk_state.reshape((1,1,8)).astype(np.float32)

            if step % 2 == 0:  # every other step to keep output short
                _, P_phys = ekf.get_state()
                P_diag = np.sqrt(np.diag(P_phys))
                print(f"  EKF √P diag (first 4): {P_diag[:4]}")
                print(f"  Innovation gain norm: {np.linalg.norm(ekf.K)}")

            # --- flatten for logging/plotting next step ---
            xk_core = (xk_state[0] if isinstance(xk_state, (list, tuple)) else xk_state)
            xk_core = xk_core.numpy() if hasattr(xk_core, "numpy") else xk_core
            xk_flat = np.asarray(xk_core).reshape(-1)   # shape (8,)
            all_states.append(xk_flat)

    finally:
        if live is not None:
            plt.close(live.fig)
            plt.ioff()  # leave interactive mode so later plt.show() works


    return np.array(all_states), np.array(all_inputs)

#  Run both simulations 
steps = 20
#X_no,  U_no = simulate(with_catalyst=False, steps=steps,Vi=Vi)
X_yes, U_yes = simulate(with_catalyst=True,  steps=steps,Vi=Vi)
time = np.arange(X_no.shape[0]) * Ts




# --- 4. PLOTTING RESULTS ----
#  Plot WITHOUT catalyst 
fig1, axes1 = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
axes1 = axes1.flatten()

for i in range(8):
    axes1[i].plot(time, X_no[:, i])
    axes1[i].set_title(f"State {i+1} (no catalyst)")
    axes1[i].set_ylabel("Value")
    axes1[i].grid(True)

for ax in axes1[-2:]:
    ax.set_xlabel("Time [s]")

plt.suptitle("System states over time - WITHOUT catalyst input", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# --- Plot WITH catalyst ---
fig2, axes2 = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
axes2 = axes2.flatten()

for i in range(8):
    axes2[i].plot(time, X_yes[:, i])
    axes2[i].set_title(f"State {i+1} (with catalyst)")
    axes2[i].set_ylabel("Value")
    axes2[i].grid(True)

for ax in axes2[-2:]:
    ax.set_xlabel("Time [s]")

plt.suptitle("System states over time - WITH catalyst input", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# --- Plot inputs (SO3 and Cl) for both cases ---
fig3, ax3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

U_no = np.asarray(U_no)
U_yes = np.asarray(U_yes)

# Inputs are applied once per control interval (Ts)
t_u_no  = np.arange(U_no.shape[0])  * Ts
t_u_yes = np.arange(U_yes.shape[0]) * Ts

# --- SO3 subplot ---
ax3[0].plot(t_u_no,  U_no[:, 0],  label='SO₃ (no catalyst case)', alpha=0.6)
ax3[0].plot(t_u_yes, U_yes[:, 0], label='SO₃ (with MPC)', linewidth=2)
ax3[0].axhline(u_max[0], color='k', linestyle='--', linewidth=1, label='SO₃ max')
ax3[0].set_ylabel('SO₃ [M]')
ax3[0].grid(True)
ax3[0].legend(loc='best')

# --- Cl subplot ---
ax3[1].plot(t_u_no,  U_no[:, 1],  label='Cl⁻ (no catalyst case)', alpha=0.6)
ax3[1].plot(t_u_yes, U_yes[:, 1], label='Cl⁻ (with MPC)', linewidth=2)
ax3[1].axhline(u_max[1], color='k', linestyle='--', linewidth=1, label='Cl⁻ max')
ax3[1].set_ylabel('Cl⁻ [M]')
ax3[1].set_xlabel('Time [s]')
ax3[1].grid(True)
ax3[1].legend(loc='best')

plt.suptitle('Catalyst inputs over time', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
