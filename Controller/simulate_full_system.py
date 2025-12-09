import argparse
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
from utils import estimate_e_with_intensity

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


# Optional CLI overrides for weights
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--qx", type=float, help="Override state cost weight qx")
parser.add_argument("--qf", type=float, help="Override terminal cost weight qf")
cli_args, _ = parser.parse_known_args()

# load parameters
params, init_vals = load_yaml_params(cfg_dir)
pH_0 = float(7.0)#float(init_vals["pH"])
c_cl_0 = float(init_vals["c_cl_0"])
c_so3_0 = float(init_vals["c_so3_0"])
intensity_0 = 0.1
#dt_sim = 5.0
k_values = load_yaml_constants(cfg_dir)
k1, k2, k3, k4, k5, k6, k7 = [k_values[f'k{i}'] for i in range(1, 8)]
initial_state = np.array([init_vals["c_pfas_init"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
initial_state = initial_state.reshape((1,1,8)).astype(np.float32) 

#cov_params = load_yaml_covariance(cfg_dir)

# Catalyst concentrations / actuation maxima (match NU=4 ordering)
c_cl = params['c_cl']  # M
c_so3 = params['c_so3']  # M
# For pH/intensity there is no stock "concentration", but we pass nominal values
# so the parameter vector matches NU=4 expected by the CasADi builder.
C_c = [c_so3, c_cl, pH_0, intensity_0]

# Catalyst limits
cl_max = c_cl*0.1
so3_max = c_so3*0.1
intensity_max = 1.0
pH_max = 12.0
u_max   = [so3_max, cl_max,pH_max, intensity_max] 

# Determine sampling time (loop time) Ts
e_max = estimate_e_with_intensity(params, c_so3=so3_max, c_cl=cl_max, pH=pH_max, c_pfas_init=init_vals["c_pfas_init"], k1=k1,intensity=intensity_max)
k_max = max([k1, k2, k3, k4, k5, k6, k7])

Ts = 49 #int(1.0 / (k_max * e_max))  # expand later to use func
dt_sim = Ts/10.0
print(f"Chosen sampling time Ts: {Ts} seconds")

# number of batches
t_r = init_vals["t_r"]
n_batches = init_vals["n_batches"]

# normalizers so costs are O(1)
x0_flat = initial_state.reshape(-1)
weights = {**DEFAULT_WEIGHTS}
if cli_args.qx is not None:
    weights["qx"] = float(cli_args.qx)
if cli_args.qf is not None:
    weights["qf"] = float(cli_args.qf)
# compute the cost of UV, it depends on Ts thus this is nessecary.
power_uv_lamp = 14
price_of_electricity = 2.60 # DKK/kWh
weights["R"][3] = power_uv_lamp * Ts * price_of_electricity /(1000*3600)   # convert to DKK per control interval

# Volume parameters
Vi = init_vals["Vi"] # initial volume
Vr = init_vals["Vr"] # reactor volume
Vmax = init_vals["Vmax"] # maximum volume
V_sens = init_vals["V_sens"] # volume sampled each step

# integration cell
rk_cell = RungeKuttaIntegratorCell(
        k1, k2, k3, k4, k5, k6, k7,
        params, c_cl_0, c_so3_0, pH_0, intensity_0, dt_sim,
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
    c_pfas_init=init_vals["c_pfas_init"],
    dt=dt_sim,
    substeps=substeps,
    N=8,
    weights=weights,
    u_max=u_max,
    x0_flat=x0_flat,             
    enable_volume_constraints=True,
    du_max=None,
    rk_cell=rk_cell,
)

# ---- 3. SIMULATE WHOLE PROCESS ----

def simulate(with_catalyst, steps, Vi):
    substeps = int(round(Ts / dt_sim))

    # --- LIVE history (for LiveMPCPlot) ---
    x0_flat = initial_state.reshape(-1)
    all_states_live = [x0_flat]      # one point per Ts, after integration
    all_times_live  = [0.0]

    # --- HYBRID history (for pretty offline plotting) ---
    all_states_plot = [x0_flat]      # will include reset + integrated states
    all_times_plot  = [0.0]
    reset_idx = []                   # indices in all_states_plot
    cont_end_idx = []                # indices in all_states_plot

    # NEW: dilution factor history (only nontrivial when with_catalyst=True)
    gamma_hist = []                  # list of gamma values
    gamma_time = []                  # time stamps where gamma is applied

    all_inputs = []
    measured_F = [0.0]

    # Include all 4 inputs (so3, cl, pH, intensity) so the MPC/TF shapes match
    uk_prev = np.array([0.0, 0.0, pH_0, intensity_0], dtype=float)
    x_scale, u_scale = make_normalizers_from_numpy(x0_flat, u_max)
    # Initialize EKF c_eaq based on initial inputs
    e_init = estimate_e_with_intensity(
        params,
        c_so3=uk_prev[0],
        c_cl=uk_prev[1],
        pH=uk_prev[2],
        intensity=uk_prev[3],
        c_pfas_init=init_vals["c_pfas_init"],
        k1=k1,
    )
    """
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
    """

    
    t_max   = steps * Ts
    z0_init = float(np.sum(initial_state.reshape(-1)[:7]))
    live = LiveMPCPlot(Ts=Ts, t_max=t_max, z0=z0_init, u_max=u_max, x0=x0_flat, make_state_grid=True) if with_catalyst else None



    try:
        for step in range(steps):
            print(f"Step {step+1}/{steps} - Current state: {all_states_live[-1]}")

            current_state = all_states_live[-1].copy()   # shape (8,)
            t_k = all_times_live[-1]
            

            if with_catalyst:
                # --- solve MPC ---
                Z_est = float(np.sum(current_state[:7]))   # ΣPFAS from EKF/plant state
                z_start = 5e-10    # start dosing above this
                z_stop  = 2e-10    # stop dosing below this for H steps
                H = 2              # hysteresis steps

                if step == 0: # Force first applied input to be exactly uk_prev (zero). we need measurement first
                    uk = uk_prev.copy()
                    Uplan = uk_prev[np.newaxis, :].repeat(ctx_adi["N"], axis=0)
                    Jstar = 0.0
                else:
                    if step >= H and all(np.sum(all_states_live[-i][:7]) < z_stop for i in range(1, H+1)):
                        uk = uk_prev
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

                    # Build safe arrays for inputs
                    if len(all_inputs):
                        t_u_hist_arr = np.arange(len(all_inputs)) * Ts
                        U_hist_arr   = np.asarray(all_inputs, dtype=float)
                    else:
                        t_u_hist_arr = np.array([], dtype=float)
                        U_hist_arr   = np.zeros((0, 2), dtype=float)

                    X_hist_arr = np.vstack(all_states_live)  # (k+1, 8)

                    live.update(
                        t_hist=np.array(all_times_live),
                        Z_hist=np.array([np.sum(x[:7]) for x in all_states_live], dtype=float),
                        t_u_hist=np.arange(len(all_inputs)) * Ts,
                        U_hist=np.asarray(all_inputs, dtype=float) if all_inputs else np.zeros((0, 2)),

                        t0_abs=step * Ts,
                        t_pred_rel=t_pred_rel,
                        Z_pred=Z_pred,
                        t_u_pred_rel=t_u_pred_rel,
                        U_plan=np.asarray(Uplan, dtype=float),

                        F_meas_t=np.arange(len(measured_F)) * Ts,
                        F_meas=np.asarray(measured_F, dtype=float),

                        X_hist=X_hist_arr,
                        X_pred=X_pred,
                    )



                # --- compute dilution like in CasADi and APPLY it to the state ---
                deltau = uk - uk_prev
                deltaC = deltau[0:2]  # change in dosed concentration

                # volume *before* dosing but after sampling, same as in CasADi: Vs = Vs - V_sens
                Vs_before = Vi - V_sens

                Vsum = 0.0
                for i in range(2):
                    Vsum += vol_from_deltaC_safe(deltaC[i], C_c[i], Vs_before, eps=1e-12)

                # dilution factor gamma = V_before / (V_before + Vsum)
                gamma = 1.0 - Vsum / (Vs_before + Vsum) if (Vs_before + Vsum) > 0 else 1.0
                gamma = float(np.clip(gamma, 0.0, 1.0))

                print("Change in volume:", Vsum)
                Vi = float(Vs_before + Vsum)      # new volume after dosing
                print("New volume:", Vi)

                # ----- APPLY dilution to the concentration state -----
                current_state = gamma * current_state   # apply reset map

                # --- store dilution factor and its time (at t_k) ---
                gamma_hist.append(gamma)
                gamma_time.append(t_k)

                # --- HYBRID history: log reset at time t_k ---
                all_states_plot.append(current_state.copy())
                all_times_plot.append(t_k)
                reset_idx.append(len(all_states_plot) - 1)
                    
            else:
                # no catalyst case
                Uplan = np.array([[0.0, 0.0, 7.0, 1.0]] * ctx_adi["N"], dtype=float)
                uk = np.array([0.0, 0.0, 7.0, 1.0], dtype=float)
            

            all_inputs.append(uk.copy())
            uk_prev = uk

              # --- advance plant by one control interval Ts (multiple Δt) ---
            # Use advance_one_control_step with n_substeps=1 repeatedly to
            # perform RK4 steps of size dt_sim and store ALL intermediate states.

            segment_states = [current_state.copy()]   # list of np arrays, each length 8
            segment_times  = [t_k]                    # corresponding times

            y_curr = current_state.copy()
            for j in range(substeps):
                # One RK4 step of size dt_sim via the existing helper
                y_tf = advance_one_control_step(
                    rk_cell,
                    y_curr.reshape(1, 1, 8).astype(np.float32),
                    uk,
                    1  # one substep = dt_sim
                )
                # y_tf is a tf.Tensor; convert to flat numpy vector
                y_np = np.reshape(y_tf[0].numpy(), (-1,))
                y_np = np.maximum(y_np, 0.0)  # avoid negative concentrations from numerical drift
                if not np.isfinite(y_np).all():
                    raise RuntimeError(f"Non-finite state encountered at step {step}, substep {j}: {y_np}")
                segment_states.append(y_np.copy())
                segment_times.append(t_k + (j + 1) * dt_sim)
                y_curr = y_np

            # Final state at end of control interval Ts
            xk_state_simulated = segment_states[-1].copy()

            # --- LIVE history: one point per Ts (sampling instants only) ---
            xk_flat = xk_state_simulated.copy()
            t_next = t_k + Ts
            all_states_live.append(xk_flat)
            all_times_live.append(t_next)

            # --- HYBRID history: full RK4 trajectory for pretty plotting ---
            # For the with_catalyst case we already appended the reset state
            # (current_state at time t_k) to all_states_plot/all_times_plot.
            # Now append ONLY the intermediate RK4 states for this interval.
            # segment_states[0] == current_state, so skip index 0.
            for s_state, s_time in zip(segment_states[1:], segment_times[1:]):
                all_states_plot.append(s_state.copy())
                all_times_plot.append(s_time)

            # mark end-of-segment index (last point in this interval)
            cont_end_idx.append(len(all_states_plot) - 1)

    finally:
        if live is not None:
            plt.close(live.fig)
            plt.ioff()  # leave interactive mode so later plt.show() works


    return (
        np.array(all_states_live),
        np.array(all_inputs),
        np.array(all_times_live),
        np.array(all_states_plot),
        np.array(all_times_plot),
        np.array(reset_idx, dtype=int),
        np.array(cont_end_idx, dtype=int),
        np.array(gamma_hist, dtype=float),
        np.array(gamma_time, dtype=float),
    )




#  Run both simulations 
steps = 12
X_no_live, U_no, t_no_live, X_no_plot, t_no_plot, reset_no, cont_no, gamma_no, t_gamma_no = simulate(with_catalyst=False, steps=steps, Vi=Vi)
X_yes_live, U_yes, t_yes_live, X_yes_plot, t_yes_plot, reset_yes, cont_yes, gamma_yes, t_gamma_yes = simulate(with_catalyst=True,  steps=steps, Vi=Vi)
# --- 4. PLOTTING RESULTS ----

import matplotlib.pyplot as plt

# -----------------------------
# Global plot style
# -----------------------------
plt.rcParams['axes.labelsize'] = 14      # axis label font size
plt.rcParams['xtick.labelsize'] = 12     # x tick font
plt.rcParams['ytick.labelsize'] = 12     # y tick font
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8

state_titles = [
    r"$x_1: \mathrm{C_7F_{15}COO^-}$",
    r"$x_2: \mathrm{C_6F_{13}COO^-}$",
    r"$x_3: \mathrm{C_5F_{11}COO^-}$",
    r"$x_4: \mathrm{C_4F_9COO^-}$",
    r"$x_5: \mathrm{C_3F_7COO^-}$",
    r"$x_6: \mathrm{C_2F_5COO^-}$",
    r"$x_7: \mathrm{CF_3COO^-}$",
    r"$x_8: \mathrm{F^-}$",
]

# ============================================================
# Find dosing steps and pre/post-dilution indices
# ============================================================
gamma_np = np.asarray(gamma_yes)
eps_gamma = 1e-12

# step indices where a real dilution happened (gamma < 1)
dose_steps = np.where(gamma_np < (1.0 - eps_gamma))[0]

# ignore step 0 (there is no "previous segment" for it)
dose_steps = dose_steps[dose_steps > 0]

reset_yes = np.asarray(reset_yes, dtype=int)
cont_yes  = np.asarray(cont_yes,  dtype=int)

# for each dosing step s:
#   pre-dilution index = cont_yes[s-1]  (end of previous segment at t = s*Ts)
#   post-dilution index = reset_yes[s]  (after dilution at same t)
pre_idx  = cont_yes[dose_steps - 1]
post_idx = reset_yes[dose_steps]

# ============================================================
# STATES: with vs. without catalyst (8x1)
# ============================================================
fig_states, axes = plt.subplots(8, 1, figsize=(10, 15), sharex=True)
axes = axes.flatten()

for i in range(8):
    ax = axes[i]

    # --- no catalyst ---
    lbl_no = "no catalyst" if i == 0 else None
    ax.plot(
        t_no_plot,
        X_no_plot[:, i],
        label=lbl_no,
        color="tab:gray",
        linewidth=2.0,
        alpha=0.9,
    )

    # --- with catalyst: base continuous curve ---
    lbl_yes = "with catalyst" if i == 0 else None
    ax.plot(
        t_yes_plot,
        X_yes_plot[:, i],
        label=lbl_yes,
        color="tab:blue",
        linewidth=2.0,
    )

    # --- draw the actual discontinuities at dosing times ---
    if dose_steps.size > 0:
        for k in range(len(dose_steps)):
            j_pre  = pre_idx[k]
            j_post = post_idx[k]

            t_dose   = t_yes_plot[j_post]           # same time for pre and post
            y_before = X_yes_plot[j_pre, i]         # before dilution
            y_after  = X_yes_plot[j_post, i]        # after dilution (= gamma*y_before)

            # blue vertical drop (same color as trajectory)
            ax.plot(
                [t_dose, t_dose],
                [y_before, y_after],
                color="tab:blue",
                linewidth=2.0,
            )

        # optional: circle markers at post-dilution points
        lbl_dose = "dose/reset" if i == 0 else None
        ax.scatter(
            t_yes_plot[post_idx],
            X_yes_plot[post_idx, i],
            s=80,
            facecolors="white",
            edgecolors="tab:blue",
            linewidths=2.0,
            marker="o",
            label=lbl_dose,
        )

    ax.set_title(state_titles[i], fontsize=14)
    ax.grid(True)

    legend = ax.legend(
        loc="upper right",
        fontsize=11,
        frameon=True,
        borderpad=0.6,
        handlelength=2.5,
        handletextpad=0.8,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

axes[-1].set_xlabel("Time [s]", fontsize=14, fontweight="bold")

fig_states.text(
    0.02, 0.5,
    "Concentration [M]",
    va="center",
    rotation="vertical",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout(rect=[0.06, 0.0, 1.0, 1.0])
plt.show()


# ============================================================
# 2) INPUTS: only MPC case, both inputs in ONE clean plot
# ============================================================

fig_u, ax_u = plt.subplots(figsize=(10, 4))

U_yes_np = np.asarray(U_yes)

# Inputs applied once per control interval
t_u_yes    = np.arange(U_yes_np.shape[0] + 1) * Ts      # 0, Ts, ..., N*Ts
U_yes_step = np.vstack([U_yes_np, U_yes_np[-1]])        # repeat last value for stairs

# Catalyst inputs (stairs)
ax_u.step(
    t_u_yes,
    U_yes_step[:, 0],
    where="post",
    label=r"SO$_3^{2-}$",
    color="tab:blue",
    linewidth=2.5,
)
ax_u.step(
    t_u_yes,
    U_yes_step[:, 1],
    where="post",
    label=r"Cl$^-$",
    color="tab:orange",
    linestyle="--",
    linewidth=2.5,
)

# Maximum constraints (thick, distinct colors)
ax_u.axhline(
    u_max[0],
    color="red",
    linestyle=":",
    linewidth=2.2,
    label=r"SO$_3^{2-}$ max",
)
ax_u.axhline(
    u_max[1],
    color="purple",
    linestyle=":",
    linewidth=2.2,
    label=r"Cl$^-$ max",
)

# Axis labels (bigger + bold)
ax_u.set_xlabel("Time [s]", fontsize=14, fontweight="bold")
ax_u.set_ylabel("Catalyst concentration [M]", fontsize=14, fontweight="bold")

ax_u.grid(True)

# Legend: bold text, background kept
legend_u = ax_u.legend(
    loc="upper right",
    fontsize=12,
    frameon=True,
    borderpad=0.6,
    handlelength=3.0,
    handletextpad=0.8,
)
for text in legend_u.get_texts():
    text.set_fontweight("bold")

plt.tight_layout()

# ============================================================
# 2b) pH and intensity over time (with catalyst)
# ============================================================
fig_pi, ax_pi = plt.subplots(figsize=(10, 4))
ax_pi.step(
    t_u_yes,
    U_yes_step[:, 2],
    where="post",
    label="pH",
    color="tab:green",
    linewidth=2.5,
)
ax_pi.step(
    t_u_yes,
    U_yes_step[:, 3],
    where="post",
    label="Intensity",
    color="tab:red",
    linestyle="--",
    linewidth=2.5,
)
ax_pi.axhline(
    u_max[2],
    color="tab:green",
    linestyle=":",
    linewidth=2.0,
    label="pH max",
)
ax_pi.axhline(
    u_max[3],
    color="tab:red",
    linestyle=":",
    linewidth=2.0,
    label="Intensity max",
)
ax_pi.set_xlabel("Time [s]", fontsize=14, fontweight="bold")
ax_pi.set_ylabel("pH / normalized intensity", fontsize=14, fontweight="bold")
ax_pi.grid(True)
legend_pi = ax_pi.legend(
    loc="upper right",
    fontsize=12,
    frameon=True,
    borderpad=0.6,
    handlelength=3.0,
    handletextpad=0.8,
)
for text in legend_pi.get_texts():
    text.set_fontweight("bold")
plt.tight_layout()

# ============================================================
# 3) Dilution factor over time (with catalyst)
# ============================================================
if gamma_yes.size > 0:
    fig_gamma, ax4 = plt.subplots(figsize=(8, 4))

    t_gamma_step = np.concatenate([t_gamma_yes, [t_gamma_yes[-1] + Ts]])
    gamma_step   = np.concatenate([gamma_yes,   [gamma_yes[-1]]])

    ax4.step(
        t_gamma_step,
        gamma_step,
        where="post",
        marker="o",
        linewidth=2.0,
    )
    ax4.set_xlabel("Time [s]", fontsize=14, fontweight="bold")
    ax4.set_ylabel(r"Dilution factor $\gamma$", fontsize=14, fontweight="bold")
    ax4.grid(True)

    plt.tight_layout()

# Finally, show everything
plt.show()

# ----------------------------------------------------------
# 4. DEGRADATION TIMES (1% of peak concentration)
# ----------------------------------------------------------

species_labels = [
    "x1 : C7F15COO-",
    "x2 : C6F13COO-",
    "x3 : C5F11COO-",
    "x4 : C4F9COO-",
    "x5 : C3F7COO-",
    "x6 : C2F5COO-",
    "x7 : CF3COO-",
    # x8 is fluoride (product), so we skip it for degradation time
]

def compute_degradation_times(t, X, case_label):
    """
    t : 1D array of times (e.g. t_no_plot or t_yes_plot)
    X : 2D array of states with shape (N, 8)
    case_label : string, e.g. "no catalyst" or "with catalyst"
    """
    print(f"\nDegradation times ({case_label})")
    print("  (time to drop to 1% of peak concentration)")
    for i in range(7):  # x1..x7 only
        xi = X[:, i]
        max_idx = np.argmax(xi)
        max_val = xi[max_idx]

        if max_val <= 0:
            print(f"  {species_labels[i]} : never forms (max = 0)")
            continue

        threshold = 0.01 * max_val

        # Look for first time *after the peak* where xi <= 1% of peak
        after_peak = xi[max_idx:]
        below = np.where(after_peak <= threshold)[0]

        if below.size == 0:
            print(f"  {species_labels[i]} : did not reach 1% of peak within simulation")
        else:
            hit_idx = max_idx + below[0]
            t_hit = t[hit_idx]
            print(f"  {species_labels[i]} : t_1% = {t_hit:.2f} s")

# Use the high-resolution trajectories for this (the *_plot arrays)
compute_degradation_times(t_no_plot,  X_no_plot,  "no catalyst")
compute_degradation_times(t_yes_plot, X_yes_plot, "with catalyst")
