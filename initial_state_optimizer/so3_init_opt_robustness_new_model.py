#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speed-focused updates:
- Use tf.function(jit_compile=True) for the heavy forward pass + objective (XLA).
- Prebuild u_batch/x0_batch/c_grid_tf for each grid and reuse them.
- Remove .numpy() debug syncs from solve loop (keep an optional DEBUG flag).
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from helper_functions import load_trained_k, load_constants, load_initials, load_cell_params

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
DT = 5.0
T_FINAL = 10000.0

PFAS_REMAINING_FRACTION = 0.10
SMOOTHNING_COEFFICIENT = 0.05

W_TIME = 0.00021866666 #0.00021866666
W_SO3  = 3.96297


SO3_MIN = 0.00
SO3_MAX = 0.01
GRID_POINTS = 40
N_REFINE = 400

T_MAX = 7000.0

N_MC = 1000
PM = 0.10
RNG_SEED = 42

ROBUST_PERTURB_MODE = "all" # "all" or "k7"

DEBUG = False   # <- set True only when debugging shapes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models_Multiple_Scripts.H_PFOA_F_ion.model import create_model

# -----------------------------------------------------------------------------
# Constants as tf.constant (avoid recreating inside compiled functions)
# -----------------------------------------------------------------------------
DT_TF = tf.constant(DT, tf.float32)
W_TIME_TF = tf.constant(W_TIME, tf.float32)
W_SO3_TF = tf.constant(W_SO3, tf.float32)

D_THR_TF = tf.constant(100.0 * (1.0 - PFAS_REMAINING_FRACTION), tf.float32)
SMOOTH_TF = tf.constant(max(1e-6, 100.0 * SMOOTHNING_COEFFICIENT), tf.float32)

# -----------------------------------------------------------------------------
# SO3 -> pH map
# -----------------------------------------------------------------------------
SO3_POINTS = np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.010], dtype=np.float32)
PH_POINTS  = np.array([5.75, 6.02, 6.205, 6.286, 6.485, 6.91], dtype=np.float32)

def SO3_to_pH(C_so3):
    C_so3 = np.asarray(C_so3, dtype=np.float32)
    if np.any((C_so3 < SO3_POINTS[0]) | (C_so3 > SO3_POINTS[-1])):
        #raise ValueError(f"C_so3 outside [{SO3_POINTS[0]}, {SO3_POINTS[-1]}]")
        C_so3 = np.asarray(C_so3, dtype=np.float32)
        return np.full_like(C_so3, 6.2, dtype=np.float32)
    return np.interp(C_so3, SO3_POINTS, PH_POINTS).astype(np.float32)

def SO3_to_pH(C_so3):
    """
    DEBUG / DUMMY:
    Freeze pH to a constant value (decouples SO3 → pH).
    """
    C_so3 = np.asarray(C_so3, dtype=np.float32)
    return np.full_like(C_so3, 6.2, dtype=np.float32)   # choose any reference pH

# -----------------------------------------------------------------------------
# u(t) builders (numpy -> tf)
# -----------------------------------------------------------------------------
def make_u_grid_batch_np(T, c_grid, c_cl, c_pfoa0):
    """
    Returns numpy u_batch with shape (G, T, 4):
      [c_cl, c_so3, pH(SO3), c_pfoa0]
    """
    c_grid = np.asarray(c_grid, dtype=np.float32)
    G = c_grid.size
    pH_vec = SO3_to_pH(c_grid).astype(np.float32)

    u = np.zeros((G, T, 4), dtype=np.float32)
    u[:, :, 0] = np.float32(c_cl)
    u[:, :, 1] = c_grid[:, None]
    u[:, :, 2] = pH_vec[:, None]
    u[:, :, 3] = np.float32(c_pfoa0)
    return u

def make_constant_u_traj(T, c_so3_value, c_cl, c_pfoa0, dtype=tf.float32):
    c_so3_value = float(c_so3_value)
    pH_value = float(SO3_to_pH(np.array([c_so3_value], np.float32))[0])

    u = np.zeros((1, T, 4), dtype=np.float32)
    u[0, :, 0] = np.float32(c_cl)
    u[0, :, 1] = np.float32(c_so3_value)
    u[0, :, 2] = np.float32(pH_value)
    u[0, :, 3] = np.float32(c_pfoa0)
    return tf.convert_to_tensor(u, dtype=dtype)

def make_init_batch(initial_states, G):
    x0 = tf.reshape(initial_states[0], (1, 8))
    return tf.repeat(x0, repeats=G, axis=0)

# -----------------------------------------------------------------------------
# Grid refinement helper
# -----------------------------------------------------------------------------
def refine_grid_from_stats(c_star_min, c_star_max, c_star_std,
                           so3_min, so3_max,
                           n_refined=60, pad_sigmas=4.0):
    c_lo = float(c_star_min) - float(pad_sigmas) * float(c_star_std)
    c_hi = float(c_star_max) + float(pad_sigmas) * float(c_star_std)
    c_lo = max(float(so3_min), c_lo)
    c_hi = min(float(so3_max), c_hi)

    if c_hi <= c_lo:
        c_mid = 0.5 * (float(c_star_min) + float(c_star_max))
        c_lo = max(float(so3_min), c_mid - 1e-4)
        c_hi = min(float(so3_max), c_mid + 1e-4)

    return np.linspace(c_lo, c_hi, int(n_refined), dtype=np.float32)

# -----------------------------------------------------------------------------
# Model builder
# -----------------------------------------------------------------------------
def build_model_with_k(cfg_dir, k_vec, t_sim, dt):
    constants = load_constants(cfg_dir)
    pH, c_cl_init, c_so3_init, c_pfas_init = load_initials(cfg_dir)

    k_vec = np.asarray(k_vec, dtype=np.float32).flatten()
    if k_vec.size == 7:
        betaj = 2.57e4
        k_cl  = 1.0e6
        k_so3 = 1.5e6
        k_vec = np.concatenate([k_vec, [betaj, k_cl, k_so3]]).astype(np.float32)

    t_sim = np.asarray(t_sim, dtype=np.float32)
    t_pinn_list = [t_sim]
    t_true_list = [t_sim[:1]]

    initial_states = np.zeros((1, 8), np.float32)
    initial_states[0, 0] = np.float32(c_pfas_init)
    initial_states = tf.convert_to_tensor(initial_states)

    model = create_model(
        *k_vec,
        constants,
        c_cl_init, c_so3_init, pH, dt,
        initial_states, t_pinn_list, t_true_list,
        for_prediction=True,
        exo_map={"c_cl": 0, "c_so3": 1, "pH": 2, "c_pfoa0": 3},
        output_mode="defluorination_pct",
    )
    model.trainable = False
    return model, initial_states, (pH, c_cl_init, c_so3_init, c_pfas_init)

# -----------------------------------------------------------------------------
# Robustness helpers
# -----------------------------------------------------------------------------
def sample_k_uniform_pm(k_nominal, pm=0.10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    k_nominal = np.asarray(k_nominal, dtype=np.float32).flatten()
    delta = rng.uniform(-pm, pm, size=k_nominal.shape).astype(np.float32)
    return (k_nominal * (1.0 + delta)).astype(np.float32)

def _get_cell(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.RNN):
            return layer.cell
    raise RuntimeError("No tf.keras.layers.RNN layer found in model.")

def set_params_on_model(model, k_vec, mode="k7"):
    cell = _get_cell(model)
    if not hasattr(cell, "log_k_values"):
        raise RuntimeError("Cell has no log_k_values; cannot set parameters in-place.")

    k_vec = np.asarray(k_vec, dtype=np.float32).flatten()

    if mode == "k7":
        names = ["k1", "k2", "k3", "k4", "k5", "k6", "k7"]
        k_use = k_vec[:7]
    elif mode == "all":
        names = ["k1","k2","k3","k4","k5","k6","k7","betaj","k_cl","k_so3"]
        if k_vec.size < 10:
            betaj = 2.57e4
            k_cl  = 1.0e6
            k_so3 = 1.5e6
            k_vec = np.concatenate([k_vec, [betaj, k_cl, k_so3]]).astype(np.float32)
        k_use = k_vec[:10]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    logk = np.log10(np.maximum(k_use, 1e-30)).astype(np.float32)
    for i, name in enumerate(names):
        if name in cell.log_k_values:
            cell.log_k_values[name].assign(logk[i])

# -----------------------------------------------------------------------------
# FAST compiled objective
# -----------------------------------------------------------------------------
@tf.function(jit_compile=True)
def eval_grid_batched_tf(model, u_batch, x0_batch, c_grid):
    """
    u_batch: (G,T,4) float32
    x0_batch: (G,8) float32
    c_grid: (G,) float32
    """
    y = model([u_batch, x0_batch], training=False)  # (G,T,1)
    defluor = y[:, :, 0]

    above = tf.nn.sigmoid((D_THR_TF - defluor) / SMOOTH_TF)  # (G,T)
    time_above = tf.reduce_sum(above, axis=1) * DT_TF        # (G,)
    cost = W_TIME_TF * time_above + W_SO3_TF * c_grid        # (G,)
    return time_above, cost

@tf.function(jit_compile=True)
def eval_single_tf(model, u_traj, x0, c_so3):
    y = model([u_traj, x0], training=False)  # (1,T,1)
    defluor = y[0, :, 0]
    above = tf.nn.sigmoid((D_THR_TF - defluor) / SMOOTH_TF)
    time_above = tf.reduce_sum(above) * DT_TF
    cost = W_TIME_TF * time_above + W_SO3_TF * c_so3
    return time_above, cost

def is_feasible(time_above):
    if T_MAX is None:
        return True
    return float(time_above) <= float(T_MAX)

# -----------------------------------------------------------------------------
# Prebuild cache for grids (THIS is a big speed win)
# -----------------------------------------------------------------------------
def prebuild_grid_tensors(T, c_grid, c_cl, c_pfoa0, initial_states):
    """
    Returns (u_tf, x0_tf, c_grid_tf) with fixed shapes.
    """
    u_np = make_u_grid_batch_np(T, c_grid, c_cl=c_cl, c_pfoa0=c_pfoa0)
    u_tf = tf.constant(u_np, dtype=tf.float32)  # constant = no host->device each call
    x0_tf = make_init_batch(initial_states, len(c_grid))
    c_tf = tf.constant(np.asarray(c_grid, dtype=np.float32), dtype=tf.float32)
    return u_tf, x0_tf, c_tf

# -----------------------------------------------------------------------------
# Solve on a prebuilt grid
# -----------------------------------------------------------------------------
def solve_grid_opt_prebuilt(model, u_tf, x0_tf, c_tf, c_grid_np):
    t_hit, cost = eval_grid_batched_tf(model, u_tf, x0_tf, c_tf)

    if T_MAX is not None:
        feasible = t_hit <= tf.constant(float(T_MAX), tf.float32)
        big = tf.constant(1e30, tf.float32)
        cost_feas = tf.where(feasible, cost, big)

        if bool(tf.reduce_any(feasible).numpy()):
            best_idx = int(tf.argmin(cost_feas).numpy())
        else:
            best_idx = int(tf.argmin(cost).numpy())
    else:
        feasible = tf.ones_like(cost, tf.bool)
        best_idx = int(tf.argmin(cost).numpy())

    best_c = float(c_grid_np[best_idx])
    best_cost = float(cost[best_idx].numpy())
    best_t = float(t_hit[best_idx].numpy())
    return best_c, best_cost, best_t, t_hit.numpy(), cost.numpy(), feasible.numpy()

def summarize(name, arr):
    arr = np.asarray(arr, dtype=float)
    return (f"{name}: mean={arr.mean():.4g}, std={arr.std():.4g}, "
            f"p05={np.percentile(arr,5):.4g}, p50={np.percentile(arr,50):.4g}, "
            f"p95={np.percentile(arr,95):.4g}, min={arr.min():.4g}, max={arr.max():.4g}")
def dump_cell_params(model):
    cell = _get_cell(model)

    out = {}

    # log_k_values -> linear
    for name, var in cell.log_k_values.items():
        out[name] = float(tf.pow(10.0, var).numpy())

    # the extra trainables created in build()
    extra = [
        "log_Kh_mM", "theta_other", "theta1", "theta2",
        "log_gamma_scale", "phi0", "phi1", "phi2",
        "log_krec"
    ]
    for name in extra:
        if hasattr(cell, name):
            v = getattr(cell, name)
            if "log_" in name:
                # these are log10
                out[name.replace("log_", "")] = float(tf.pow(10.0, v).numpy())
            else:
                out[name] = float(v.numpy())
        else:
            out[name] = None

    return out

def set_cell_extra_params(model, d):
    """
    d contains *linear-space* values:
      Kh_mM, gamma_scale, krec in linear units
      thetas/phis in their native units (no transform)
    """
    cell = _get_cell(model)

    def _as_float(key):
        if key not in d:
            return None
        return float(d[key])

    # --- direct params ---
    for name in ["theta_other", "theta1", "theta2", "phi0", "phi1", "phi2"]:
        v = _as_float(name)
        if v is not None and hasattr(cell, name):
            getattr(cell, name).assign(np.float32(v))

    # --- log10 params ---
    log_map = {
        "Kh_mM": "log_Kh_mM",
        "gamma_scale": "log_gamma_scale",
        "krec": "log_krec",
    }
    for lin_name, log_name in log_map.items():
        v = _as_float(lin_name)
        if v is None:
            continue
        if v <= 0:
            raise ValueError(f"{lin_name} must be > 0, got {v}")
        if hasattr(cell, log_name):
            getattr(cell, log_name).assign(np.float32(np.log10(v)))

def get_cell_param_dict(model):
    """
    Returns a dict in *linear space* for everything we care about.
    (k's in linear, Kh_mM/gamma_scale/krec in linear, theta/phi raw)
    """
    cell = _get_cell(model)
    out = {}

    # k1..k7, betaj, k_cl, k_so3 (stored as log10)
    for name, var in cell.log_k_values.items():
        out[name] = float(10.0 ** var.numpy())

    # extra log10 params -> linear
    out["Kh_mM"]       = float(10.0 ** cell.log_Kh_mM.numpy())
    out["gamma_scale"] = float(10.0 ** cell.log_gamma_scale.numpy())
    out["krec"]        = float(10.0 ** cell.log_krec.numpy())

    # linear params
    for name in ["theta_other","theta1","theta2","phi0","phi1","phi2"]:
        out[name] = float(getattr(cell, name).numpy())

    return out


def set_cell_param_dict(model, d):
    """
    Assign from a dict in *linear space*.
    Handles internal log10 variables correctly.
    """
    cell = _get_cell(model)

    # 1) k1..k7, betaj, k_cl, k_so3 (log10 vars)
    for name, var in cell.log_k_values.items():
        if name in d:
            v = float(d[name])
            if v <= 0:
                raise ValueError(f"{name} must be >0, got {v}")
            var.assign(np.float32(np.log10(v)))

    # 2) extra log10 vars
    for lin_name, log_name in [("Kh_mM","log_Kh_mM"), ("gamma_scale","log_gamma_scale"), ("krec","log_krec")]:
        if lin_name in d and hasattr(cell, log_name):
            v = float(d[lin_name])
            if v <= 0:
                raise ValueError(f"{lin_name} must be >0, got {v}")
            getattr(cell, log_name).assign(np.float32(np.log10(v)))

    # 3) linear vars
    for name in ["theta_other","theta1","theta2","phi0","phi1","phi2"]:
        if name in d and hasattr(cell, name):
            getattr(cell, name).assign(np.float32(float(d[name])))


def perturb_param_dict(d_nom, pm, rng):
    """
    Multiplicative perturbation for positive params.
    Additive perturbation for signed params.
    """
    d = dict(d_nom)

    # positive params: multiply by (1+eps)
    positive = ["k1","k2","k3","k4","k5","k6","k7","betaj","k_cl","k_so3",
                "Kh_mM","gamma_scale","krec"]
    for k in positive:
        eps = rng.uniform(-pm, pm)
        d[k] = float(d[k]) * (1.0 + float(eps))

    # signed params: additive (scale by magnitude to be unit-consistent)
    signed = ["theta_other","theta1","theta2","phi0","phi1","phi2"]
    for k in signed:
        base = float(d[k])
        scale = max(1e-6, abs(base))     # if base ~0, still perturb a tiny amount
        eps = rng.uniform(-pm, pm)
        d[k] = base + float(eps) * scale

    return d

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "config"
    trained_k_yaml = cfg_dir / "trained_params.yaml"

    t_sim = np.arange(0.0, T_FINAL, DT, dtype=np.float32)
    T = t_sim.size

    c_so3_grid_coarse = np.linspace(SO3_MIN, SO3_MAX, GRID_POINTS, dtype=np.float32)

    k_nom = load_trained_k(trained_k_yaml)

    model_nom, init_nom, (pH, c_cl_init, c_so3_init, c_pfas_init) = build_model_with_k(cfg_dir, k_nom, t_sim, DT)
    model_nom.trainable = False
    cell_params = load_cell_params(cfg_dir / "trained_cell_params.yaml")
    set_cell_extra_params(model_nom, cell_params)
    cell_nom = get_cell_param_dict(model_nom)
    cell = _get_cell(model_nom)

    # Warm up compile once (important)
    u_warm = make_constant_u_traj(T, float(SO3_MIN), c_cl_init, c_pfas_init)
    _ = model_nom([u_warm, init_nom], training=False)


    # Prebuild coarse tensors
    u_coarse, x0_coarse, c_coarse_tf = prebuild_grid_tensors(
        T, c_so3_grid_coarse, c_cl=c_cl_init, c_pfoa0=c_pfas_init, initial_states=init_nom
    )

    if DEBUG:
        u0 = u_coarse[:, 0, :]
        c_cl_, c_so3_, pH_, c_pfoa0_ = cell.debug_extract(u0)
        print("extracted c_so3 min/max:", float(tf.reduce_min(c_so3_)), float(tf.reduce_max(c_so3_)))
        print("extracted pH  min/max:", float(tf.reduce_min(pH_)), float(tf.reduce_max(pH_)))

    # Coarse solve (fast)
    c_nom_coarse, J_nom_coarse, t_nom_coarse, t_hist_coarse, J_hist_coarse, feas_coarse = solve_grid_opt_prebuilt(
        model_nom, u_coarse, x0_coarse, c_coarse_tf, c_so3_grid_coarse
    )
    
    print(f"[coarse] c*={c_nom_coarse:.6e}, J*={J_nom_coarse:.6e}, time_above={t_nom_coarse:.3g}s")

    # Refined grid
    c_star_std = (SO3_MAX-SO3_MIN)/GRID_POINTS
    c_star_min = c_nom_coarse - 2 * c_star_std
    c_star_max = c_nom_coarse + 2 * c_star_std
    c_so3_grid_refined = refine_grid_from_stats(
        c_star_min=c_star_min, c_star_max=c_star_max, c_star_std=c_star_std,
        so3_min=SO3_MIN, so3_max=SO3_MAX, n_refined=N_REFINE, pad_sigmas=4.0
    )
    print(f"Refined grid: [{c_so3_grid_refined[0]:.6e}, {c_so3_grid_refined[-1]:.6e}] with {len(c_so3_grid_refined)} points")

    # Prebuild refined tensors
    u_ref, x0_ref, c_ref_tf = prebuild_grid_tensors(
        T, c_so3_grid_refined, c_cl=c_cl_init, c_pfoa0=c_pfas_init, initial_states=init_nom
    )

    # Refined solve
    c_nom, J_nom, t_nom, t_hist_nom, J_hist_nom, feas_nom = solve_grid_opt_prebuilt(
        model_nom, u_ref, x0_ref, c_ref_tf, c_so3_grid_refined
    )
    print(f"[refined] Nominal optimum: c*={c_nom:.6e}, J*={J_nom:.6e}, time_above={t_nom:.3g}s")
    if T_MAX is not None:
        print(f"Nominal feasibility (T_MAX={T_MAX}s): {is_feasible(t_nom)}")

    # Nominal u(t) at optimum (single)
    u_nom = make_constant_u_traj(T, c_nom, c_cl_init, c_pfas_init)
    c_nom_tf = tf.constant(np.float32(c_nom), tf.float32)

    # Robustness loop
    rng = np.random.default_rng(RNG_SEED)
    c_star_samples, J_star_samples, t_star_samples, feasible_star = [], [], [], []
    J_at_nom_samples, t_at_nom_samples, feasible_at_nom = [], [], []

    print(f"\nRobustness: N={N_MC}, uniform ±{PM*100:.0f}% perturbation, mode={ROBUST_PERTURB_MODE}")
    for s in range(N_MC):
        #k_s = sample_k_uniform_pm(k_nom, pm=PM, rng=rng)
        d_s = perturb_param_dict(cell_nom, PM, rng)
        set_cell_param_dict(model_nom, d_s)

        c_s, J_s, t_s, _, _, _ = solve_grid_opt_prebuilt(model_nom, u_ref, x0_ref, c_ref_tf, c_so3_grid_refined)
        c_star_samples.append(c_s); J_star_samples.append(J_s); t_star_samples.append(t_s); feasible_star.append(is_feasible(t_s))

        t_nom_s_tf, J_nom_s_tf = eval_single_tf(model_nom, u_nom, init_nom, c_nom_tf)
        t_nom_s = float(t_nom_s_tf.numpy())
        J_nom_s = float(J_nom_s_tf.numpy())
        t_at_nom_samples.append(t_nom_s); J_at_nom_samples.append(J_nom_s); feasible_at_nom.append(is_feasible(t_nom_s))

        print(f"sample {s+1:03d}: c*_s={c_s:.3e}, J*_s={J_s:.3e}, t*_s={t_s:.3g}s | @c_nom: J={J_nom_s:.3e}, t={t_nom_s:.3g}s")

    # Summaries
    c_star_samples = np.array(c_star_samples, dtype=float)
    J_star_samples = np.array(J_star_samples, dtype=float)
    t_star_samples = np.array(t_star_samples, dtype=float)
    feasible_star = np.array(feasible_star, dtype=bool)

    J_at_nom_samples = np.array(J_at_nom_samples, dtype=float)
    t_at_nom_samples = np.array(t_at_nom_samples, dtype=float)
    feasible_at_nom = np.array(feasible_at_nom, dtype=bool)

    print("\n--- Robustness summary ---")
    print(summarize("c*_s", c_star_samples))
    print(summarize("J*_s", J_star_samples))
    print(summarize("t*_s [s]", t_star_samples))
    print(summarize("J(c_nom)_s", J_at_nom_samples))
    print(summarize("t(c_nom)_s [s]", t_at_nom_samples))

    if T_MAX is not None:
        print(f"Feasible rate (re-optimized): {feasible_star.mean()*100:.1f}%")
        print(f"Feasible rate (nominal c_nom): {feasible_at_nom.mean()*100:.1f}%")
        print(f"Violations @ c_nom: {np.sum(~feasible_at_nom)}/{N_MC}")
    
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size" : 16,
        "axes.labelsize": 16,     # x/y labels
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,

    })

    # Plots (unchanged)
    plt.figure()
    plt.plot(c_so3_grid_refined, J_hist_nom, marker="o")
    plt.axvline(c_nom, linestyle="--")
    plt.xlabel("C_SO3 [M]")
    plt.ylabel("Nominal cost J")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.hist(c_star_samples, bins=9)
    plt.xlabel("Optimal C_SO3 under perturbed params (c*_s)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Wahtever.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(c_so3_grid_coarse, J_hist_coarse,color="black", marker="o")
    plt.axvline(c_nom, linestyle="--",color="black")
    plt.xlabel("$C_{\mathrm{SO}_3}$ [M]")
    plt.ylabel("Nominal cost $\mathcal{J}$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Cost_function.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.hist(J_at_nom_samples, bins=15,color="black",edgecolor="black")
    plt.xlabel("$\mathcal{J}(C_{\mathrm{SO}_3}^\star)$")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Cost_hist.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.hist(t_at_nom_samples, bins=15,color="black",edgecolor="black")
    plt.xlabel("$\hat{T}_\Omega(C_{\mathrm{SO}_3}^\star)$ [s]")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Terminal_time_hist.png", dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()