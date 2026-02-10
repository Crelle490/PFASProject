import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

# --- Hardcoded simulation/optimization settings ---
DT = 5.0
T_FINAL = 700.0

PFAS_THRESHOLD = 1e-10
THRESHOLD_SMOOTHING = 1e-9

W_TIME = 0.00021866666  # DKK/s @393.6W and 2 kr/kWh
W_SO3  = 3.96297        # DKK*L/mol @90ml equivalent to one batch through reactor

SO3_MIN = 0.0
SO3_MAX = 0.005
GRID_POINTS = 30

# Optional "efficiency" constraint (set to None to disable)
T_MAX = 800.0  # seconds (or None)

# Robustness settings (Option A)
N_MC = 50       # 20-50 is usually enough for a paper
PM = 0.10       # ±10% parameter perturbation
RNG_SEED = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models_Multiple_Scripts.E_TF_MultipleBatch_Adaptive_c.model import create_model

@tf.function
def eval_grid_batched_tf(model, u_batch, x0_batch, c_grid):
    """
    All TensorFlow ops. Runs as a compiled graph.
    """
    y = model([u_batch, x0_batch], training=False)      # (G,T,8)

    total_pfas = tf.reduce_sum(y[:, :, :7], axis=-1)    # (G,T)
    above = tf.nn.sigmoid((total_pfas - PFAS_THRESHOLD) / THRESHOLD_SMOOTHING)

    time_above = tf.reduce_sum(above, axis=1) * DT      # (G,)
    cost = W_TIME * time_above + W_SO3 * c_grid          # (G,)

    return time_above, cost

def load_trained_k(path):
    d = yaml.safe_load(open(path, "r"))
    keys = [f"k{i}" for i in range(1, 8)]
    return np.array([d[k] for k in keys], dtype=np.float32)


def load_constants(cfg_dir):
    return yaml.safe_load(open(Path(cfg_dir) / "physichal_paramters.yaml", "r"))


def load_initials(cfg_dir):
    d = yaml.safe_load(open(Path(cfg_dir) / "initial_conditions.yaml", "r"))
    c_cl = d.get("c_cl", d.get("c_cl_0"))
    c_so3 = d.get("c_so3", d.get("c_so3_0"))
    if c_cl is None or c_so3 is None:
        raise KeyError("initial_conditions.yaml must define c_cl/c_so3 or c_cl_0/c_so3_0")
    return float(d["pH"]), float(c_cl), float(c_so3), float(d["c_pfas_init"])


def make_constant_u_traj(T, c_so3_value, dtype=tf.float32):
    """u(t) = [c_so3(t)] with shape (1, T, 1)"""
    u = np.zeros((1, T, 1), dtype=np.float32)
    u[0, :, 0] = np.float32(c_so3_value)
    return tf.convert_to_tensor(u, dtype=dtype)


def build_model_with_k(cfg_dir, k_vec, t_sim, dt):
    """Build model for a given parameter vector k_vec (no training)."""
    constants = load_constants(cfg_dir)
    pH, c_cl_init, c_so3_init, c_pfas_init = load_initials(cfg_dir)

    t_sim = np.asarray(t_sim, dtype=np.float32)
    t_pinn_list = [t_sim]
    t_true_list = [t_sim[:1]]

    initial_states = np.zeros((1, 8), np.float32)
    initial_states[0, 0] = np.float32(c_pfas_init)
    initial_states = tf.convert_to_tensor(initial_states)

    model = create_model(*k_vec, constants,
                         c_cl_init, c_so3_init, pH, dt,
                         initial_states, t_pinn_list, t_true_list,
                         for_prediction=True)
    model.trainable = False
    return model, initial_states


def eval_time_above_and_cost(model, initial_states, u_traj, c_so3_value):
    """
    Returns:
      time_above: soft count of time above PFAS threshold (s)
      cost: J = W_TIME*time_above + W_SO3*c_so3
    """
    y_pred = model([u_traj, initial_states], training=False)

    # keep consistent with your "PFAS states" definition.
    # You used :6; if your PFAS species are x1..x7, consider :7 instead.
    total_pfas = tf.reduce_sum(y_pred[:, :, :7], axis=-1)

    above = tf.nn.sigmoid((total_pfas - PFAS_THRESHOLD) / THRESHOLD_SMOOTHING)

    time_above = float(tf.reduce_sum(above) * DT)
    cost = float(W_TIME * time_above + W_SO3 * float(c_so3_value))
    return time_above, cost


def is_feasible(time_above):
    """Efficiency constraint: reach threshold quickly enough."""
    if T_MAX is None:
        return True
    return time_above <= float(T_MAX)


def sample_k_uniform_pm(k_nominal, pm=0.10, rng=None):
    """Independent uniform perturbation for each k_j: k_j*(1+eps), eps~U[-pm, pm]."""
    if rng is None:
        rng = np.random.default_rng()
    delta = rng.uniform(-pm, pm, size=k_nominal.shape).astype(np.float32)
    return (k_nominal * (1.0 + delta)).astype(np.float32)


def solve_grid_opt_batched(model, initial_states, t_sim, c_so3_grid):
    T = t_sim.size
    G = len(c_so3_grid)

    u_batch = make_u_grid_batch(T, c_so3_grid)                  # (G,T,1)
    x0_batch = make_init_batch(initial_states, G)               # (G,8)
    c_grid_tf = tf.convert_to_tensor(c_so3_grid, tf.float32)    # (G,)

    # 🔥 compiled execution happens here
    time_above, cost = eval_grid_batched_tf(
        model, u_batch, x0_batch, c_grid_tf
    )

    # Everything below can stay Python (cheap)
    if T_MAX is not None:
        feasible = time_above <= float(T_MAX)
        big = tf.constant(1e30, tf.float32)
        cost_feas = tf.where(feasible, cost, big)

        if tf.reduce_any(feasible):
            best_idx = int(tf.argmin(cost_feas).numpy())
        else:
            best_idx = int(tf.argmin(cost).numpy())
    else:
        feasible = tf.ones_like(cost, tf.bool)
        best_idx = int(tf.argmin(cost).numpy())

    best_c = float(c_so3_grid[best_idx])
    best_cost = float(cost[best_idx].numpy())
    best_t = float(time_above[best_idx].numpy())

    return (
        best_c,
        best_cost,
        best_t,
        time_above.numpy(),
        cost.numpy(),
        feasible.numpy()
    )


def summarize(name, arr):
    arr = np.asarray(arr, dtype=float)
    return (f"{name}: mean={arr.mean():.4g}, std={arr.std():.4g}, "
            f"p05={np.percentile(arr,5):.4g}, p50={np.percentile(arr,50):.4g}, "
            f"p95={np.percentile(arr,95):.4g}, min={arr.min():.4g}, max={arr.max():.4g}")

def set_k_on_model(model, k_vec):
    """
    Update k1..k7 in-place (no rebuild).
    k_vec shape (7,) in linear space.
    """
    rnn_layer = model.get_layer("rnn")
    cell = rnn_layer.cell  # RungeKuttaIntegratorCell

    k_vec = np.asarray(k_vec, dtype=np.float32)
    logk = np.log10(np.maximum(k_vec, 1e-30)).astype(np.float32)

    for i, name in enumerate(["k1","k2","k3","k4","k5","k6","k7"]):
        cell.log_k_values[name].assign(logk[i])

def make_u_grid_batch(T, c_grid):
    """
    Returns u_batch with shape (G, T, 1)
    """
    c_grid = np.asarray(c_grid, dtype=np.float32)
    G = c_grid.size
    u = np.zeros((G, T, 1), dtype=np.float32)
    u[:, :, 0] = c_grid[:, None]
    return tf.convert_to_tensor(u)

def make_init_batch(initial_states, G):
    """
    Repeat initial state to match batch size.
    initial_states: (1,8) tensor
    returns: (G,8)
    """
    x0 = tf.reshape(initial_states[0], (1, 8))
    return tf.repeat(x0, repeats=G, axis=0)

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "config"
    trained_k_yaml = cfg_dir / "trained_params.yaml"

    t_sim = np.arange(0.0, T_FINAL, DT, dtype=np.float32)
    c_so3_grid = np.linspace(SO3_MIN, SO3_MAX, GRID_POINTS, dtype=np.float32)

    # -----------------------------
    # Nominal optimization (once)
    # -----------------------------
    k_nom = load_trained_k(trained_k_yaml)
    model_nom, init_nom = build_model_with_k(cfg_dir, k_nom, t_sim, DT)
    model_nom.trainable = False

    # after model_nom, init_nom are created
    T = t_sim.size
    _ = model_nom([make_constant_u_traj(T, float(SO3_MIN)), init_nom], training=False)

    # nominal optimization (batched)
    c_nom, J_nom, t_nom, t_hist_nom, J_hist_nom, feas_nom = solve_grid_opt_batched(
        model_nom, init_nom, t_sim, c_so3_grid
    )

    print(f"Nominal optimum: c*={c_nom:.6e}, J*={J_nom:.6e}, time_above={t_nom:.3g}s")
    if T_MAX is not None:
        print(f"Nominal feasibility (T_MAX={T_MAX}s): {is_feasible(t_nom)}")

    # Evaluate nominal optimum under perturbed models (robustness of chosen operating point)
    T = t_sim.size
    u_nom = make_constant_u_traj(T, c_nom)

    # -----------------------------
    # Option A: Parametric sensitivity robustness check
    # - For each perturbed k: re-solve the optimization
    # - Also evaluate the nominal optimum under that perturbed k
    # -----------------------------
    rng = np.random.default_rng(RNG_SEED)

    c_star_samples = []
    J_star_samples = []
    t_star_samples = []
    feasible_star = []

    J_at_nom_samples = []
    t_at_nom_samples = []
    feasible_at_nom = []

    print(f"\nOption A robustness: N={N_MC}, k_j uniform in ±{PM*100:.0f}% around nominal")
    for s in range(N_MC):
        k_s = sample_k_uniform_pm(k_nom, pm=PM, rng=rng)

        # update parameters in-place (no rebuild)
        set_k_on_model(model_nom, k_s)

        # (A1) re-optimize under perturbed k (batched)
        c_s, J_s, t_s, _, _, _ = solve_grid_opt_batched(model_nom, init_nom, t_sim, c_so3_grid)
        c_star_samples.append(c_s)
        J_star_samples.append(J_s)
        t_star_samples.append(t_s)
        feasible_star.append(is_feasible(t_s))

        # (A2) Evaluate the nominal optimum under this perturbed model
        t_nom_s, J_nom_s = eval_time_above_and_cost(model_nom, init_nom, u_nom, c_nom)
        t_at_nom_samples.append(t_nom_s)
        J_at_nom_samples.append(J_nom_s)
        feasible_at_nom.append(is_feasible(t_nom_s))

        print(f"sample {s+1:03d}: "
              f"c*_s={c_s:.3e}, J*_s={J_s:.3e}, t*_s={t_s:.3g}s | "
              f"@c_nom: J={J_nom_s:.3e}, t={t_nom_s:.3g}s")

    c_star_samples = np.array(c_star_samples)
    J_star_samples = np.array(J_star_samples)
    t_star_samples = np.array(t_star_samples)
    feasible_star = np.array(feasible_star, dtype=bool)

    J_at_nom_samples = np.array(J_at_nom_samples)
    t_at_nom_samples = np.array(t_at_nom_samples)
    feasible_at_nom = np.array(feasible_at_nom, dtype=bool)

    # -----------------------------
    # Metrics to report in paper
    # -----------------------------
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

    # Useful robustness deltas
    delta_c = c_star_samples.max() - c_star_samples.min()
    worst_cost_increase_at_nom = J_at_nom_samples.max() - J_nom
    worst_time_at_nom = t_at_nom_samples.max()

    print(f"\nΔc* spread = {delta_c:.3e} (max-min)")
    print(f"Worst-case cost increase at nominal optimum: {worst_cost_increase_at_nom:.3e}")
    if T_MAX is not None:
        print(f"Worst-case time_above at nominal optimum: {worst_time_at_nom:.3g}s (T_MAX={T_MAX})")
    else:
        print(f"Worst-case time_above at nominal optimum: {worst_time_at_nom:.3g}s")

    # -----------------------------
    # Plots (simple, paper-friendly)
    # -----------------------------
    plt.figure()
    plt.plot(c_so3_grid, J_hist_nom, marker="o")
    plt.axvline(c_nom, linestyle="--")
    plt.xlabel("C_SO3 [M]")
    plt.ylabel("Nominal cost J")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.hist(c_star_samples, bins=15)
    plt.xlabel("Optimal C_SO3 under perturbed k (c*_s)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.hist(J_at_nom_samples, bins=15)
    plt.xlabel("Cost under perturbed k at nominal optimum J(c_nom)_s")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.hist(t_at_nom_samples, bins=15)
    plt.xlabel("time_above under perturbed k at nominal optimum t(c_nom)_s [s]")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()