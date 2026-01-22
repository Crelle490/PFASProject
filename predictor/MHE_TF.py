"""
demo_mhe_hpinn.py

Demo script showing how to run the timestamped MHE that uses an HPINN rollout
to build one-step predictions x_{i+1}^hat = f(x_i, u_i, t_i, t_{i+1}).

What this demo does:
1) Creates an MHE instance (your class).
2) Generates synthetic "ground truth" using the SAME HPINN model (so the demo is consistent).
3) Samples measurements at irregular timestamps + adds relative noise.
4) Feeds (y_k, t_k, u_{k-1}) to the MHE.
5) Plots estimated vs measured output (and optionally states).

Notes:
- This assumes your project has predictor/ode_runtime.py and config/trained_params.yaml.
- Your HPINN expects an 8D state. Your MHE uses n=8 in this demo to avoid 7↔8 mapping issues.
- If you truly want n=7, you must define how to build the 8th state from the 7 estimated ones.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from collections import deque
import pandas as pd
import tensorflow_probability as tfp
import json
from datetime import datetime

# -----------------------------
# Project import setup
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.ode_runtime import build_model_from_config

# -----------------------------
# Settings
# -----------------------------
DT = 10.0          # HPINN internal dt

# -----------------------------
# Your MHE class (as provided, with small fixes)
# -----------------------------
class MovingHorizonEstimator:
    def __init__(
        self,
        n: int,
        sampling_period: int,
        verbose: int = 2,
        horizon: int = 10,
        Q_diag: float | np.ndarray = 3.69e-16,
        R_rel: float = 1e-4,
        P0_diag: float | np.ndarray = 1.0,
        enforce_nonneg: bool = True,
        use_log_measurement: bool = True,
        eps_y: float = 1e-12,
        max_nfev: int = 300
    ):
        self.n_simulation_steps = int(sampling_period / DT) + 1
        self.model = self.HPINN_model(n_simulation_steps=self.n_simulation_steps)

        self.dummy_input_for_model = tf.convert_to_tensor(
            np.zeros((1, self.n_simulation_steps, 1), dtype=np.float32),
            dtype=tf.float32
        )
        self.verbose = int(verbose)
        self.n = int(n)
        self.N = int(horizon)

        self.Q = np.ones(self.n) * float(Q_diag) if np.isscalar(Q_diag) else np.array(Q_diag, float).reshape(self.n)
        self.P0 = np.ones(self.n) * float(P0_diag) if np.isscalar(P0_diag) else np.array(P0_diag, float).reshape(self.n)
        self.R_rel = float(R_rel)

        self.enforce_nonneg = bool(enforce_nonneg)
        self.use_log_measurement = bool(use_log_measurement)
        self.eps_y = float(eps_y)
        self.max_nfev = int(max_nfev)

        self.y_buf = deque(maxlen=self.N + 1)
        self.u_buf = deque(maxlen=self.N)
        self.t_buf = deque(maxlen=self.N + 1)

        self._x_seq_last = None
        self.x_prior = np.ones(self.n) * 1e-6

        self._res_call_count = 0

    def set_prior(self, x_prior: np.ndarray):
        self.x_prior = np.array(x_prior, dtype=float).reshape(self.n)

    def _pack(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(-1)
    
    @tf.function
    def _softplus_pos(self, z_u, eps=1e-12):
        return tf.nn.softplus(z_u) + tf.cast(eps, z_u.dtype)

    @tf.function
    def _unpack_M_tf(self, z, M):
        # z shape: ((M+1)*n,)
        return tf.reshape(z, (M + 1, self.n))

    @tf.function
    def _h_tf(self, X):
        # X: (M+1, n) -> y_hat: (M+1,)
        # measurement is last state component
        return X[:, -1]
    
    @tf.function
    def _f_batch_tf(self, X0_batch, time_starts, time_ends):
        """
        TF version of f_batch:
          X0_batch: (B, 8) float32
          time_starts/time_ends: (B,) float32
          returns: (B, 8) float32
        """
        B = tf.shape(X0_batch)[0]
        dummy = tf.zeros((B, self.n_simulation_steps, 1), dtype=tf.float32)

        # model returns (B, T, 8)
        x_series = self.model([dummy, X0_batch], training=False)

        # prepend x0: (B, T+1, 8)
        x_full = tf.concat([X0_batch[:, None, :], x_series], axis=1)

        dt = tf.cast(time_ends - time_starts, tf.float32)
        idx = tf.cast(tf.round(dt / tf.constant(DT, tf.float32)), tf.int32)
        idx = tf.clip_by_value(idx, 0, tf.shape(x_full)[1] - 1)

        # gather x_full[b, idx[b], :]
        b_idx = tf.range(B, dtype=tf.int32)
        gather_idx = tf.stack([b_idx, idx], axis=1)  # (B,2)
        X_plus = tf.gather_nd(x_full, gather_idx)    # (B,8)

        return X_plus
    
    @tf.function
    def _residuals_M_tf(self, z_u, M, y_buf_tf, t_buf_tf):
        """
        Build residual vector r(z) with TF ops.
        Inputs:
          z_u: unconstrained decision vector ( (M+1)*n, )
          M: python int or tf int
          y_buf_tf: (M+1,) float32  [latest window]
          t_buf_tf: (M+1,) float32  [latest window]
        Returns:
          r: (n + M*n + (M+1),) float32  (arrival + dyn + meas)
        """
        # positivity constraint
        z = self._softplus_pos(z_u, eps=self.eps_y)

        X = self._unpack_M_tf(z, M)  # (M+1, n)

        # arrival
        x_prior_tf = tf.convert_to_tensor(self.x_prior, tf.float32)
        P0_tf      = tf.convert_to_tensor(self.P0, tf.float32)
        r_arr = (X[0] - x_prior_tf) / tf.sqrt(P0_tf)  # (n,)

        # dynamics
        if M > 0:
            time_starts = t_buf_tf[:-1]
            time_ends   = t_buf_tf[1:]
            X0_batch    = X[:-1]  # (M,n)

            X_next_hat = self._f_batch_tf(tf.cast(X0_batch, tf.float32),
                                          tf.cast(time_starts, tf.float32),
                                          tf.cast(time_ends, tf.float32))  # (M,n)

            Q_tf = tf.convert_to_tensor(self.Q, tf.float32)
            r_dyn = (X[1:] - X_next_hat) / tf.sqrt(Q_tf)   # (M,n)
            r_dyn = tf.reshape(r_dyn, (-1,))              # (M*n,)
        else:
            r_dyn = tf.zeros((0,), tf.float32)

        # measurement residuals
        y_hat = self._h_tf(X)  # (M+1,)

        if self.use_log_measurement:
            y_safe    = tf.maximum(y_buf_tf, tf.cast(self.eps_y, tf.float32))
            yhat_safe = tf.maximum(y_hat,    tf.cast(self.eps_y, tf.float32))
            r_meas = (tf.math.log(y_safe) - tf.math.log(yhat_safe)) / tf.cast(self.R_rel, tf.float32)
        else:
            sigma = tf.cast(self.R_rel, tf.float32) * tf.maximum(tf.abs(y_buf_tf), tf.cast(self.eps_y, tf.float32))
            r_meas = (y_buf_tf - y_hat) / sigma

        # concat
        return tf.concat([tf.cast(r_arr, tf.float32), tf.cast(r_dyn, tf.float32), tf.cast(r_meas, tf.float32)], axis=0)

    def _initial_guess_M(self, M: int) -> np.ndarray:
        if self._x_seq_last is None:
            X0 = np.tile(self.x_prior, (M + 1, 1))
            return self._pack(X0)

        X_prev = self._x_seq_last  # shape (prev_M+1, n)
        prev_M = X_prev.shape[0] - 1

        if prev_M == M:
            # Same size warm start: shift and hold last
            X0 = np.vstack([X_prev[1:], X_prev[-1:]])
        elif prev_M > M:
            # Shrink: take last M+1 states of shifted trajectory
            X_shift = np.vstack([X_prev[1:], X_prev[-1:]])
            X0 = X_shift[-(M + 1):]
        else:
            # Grow: shift then pad by repeating last
            X_shift = np.vstack([X_prev[1:], X_prev[-1:]])
            pad = np.tile(X_shift[-1:], (M + 1 - X_shift.shape[0], 1))
            X0 = np.vstack([X_shift, pad])

        return self._pack(X0)


    def _initial_guess(self) -> np.ndarray:
        if self._x_seq_last is None:
            X0 = np.tile(self.x_prior, (self.N + 1, 1))
            return self._pack(X0)
        X_prev = self._x_seq_last
        X0 = np.vstack([X_prev[1:], X_prev[-1:]])  # shift warm-start
        return self._pack(X0)



    def _residuals(self, z: np.ndarray) -> np.ndarray:
        X = self._unpack(z)  # (N+1, n)

        self._res_call_count += 1
        debug = (self._res_call_count % 50 == 1)  # every 50 calls

        if debug:
            print(f"[RES] call {self._res_call_count}")
            print("  X[0] min/max:", X[0].min(), X[0].max())

        # Arrival cost
        r_arr = (X[0] - self.x_prior) / np.sqrt(self.P0)

        t = np.asarray(list(self.t_buf), dtype=float)  # length N+1
        time_starts = t[:-1]
        time_ends   = t[1:]

        X0_batch    = X[:self.N, :]  # (N, 8)

        X_next_hat = self.f_batch(X0_batch, time_starts, time_ends)  # (N, 8)

        # r_dyn uses X_next_hat[i] for each i
        r_dyn = (X[1:self.N+1] - X_next_hat) / np.sqrt(self.Q)
        r_dyn = r_dyn.reshape(-1)


        # Measurement residuals
        r_meas = []
        for i in range(self.N + 1):
            y_i = float(self.y_buf[i])
            y_hat = float(self.h(X[i]))

            if self.use_log_measurement:
                y_safe = max(y_i, self.eps_y)
                yhat_safe = max(y_hat, self.eps_y)
                r = (np.log(y_safe) - np.log(yhat_safe)) / self.R_rel
            else:
                sigma = self.R_rel * max(abs(y_i), self.eps_y)
                r = (y_i - y_hat) / sigma

            r_meas.append([r])

        r_meas = np.array(r_meas, dtype=float).reshape(-1)
        if debug:
            print("  r_dyn norm:", np.linalg.norm(r_dyn))
            print("  r_meas norm:", np.linalg.norm(r_meas))

        return np.concatenate([r_arr, r_dyn, r_meas])
    
    def log(self, level: int, *msg):
        if self.verbose >= level:
            print(*msg, flush=True)


    def update(self, y_new: float, t_new: float, u_new: float | None = None) -> dict:
        self.log(1, f"[UPDATE] t={t_new:.2f}, y={y_new:.3e}, "
            f"len(y)={len(self.y_buf)+1}/{self.N+1}, len(t)={len(self.t_buf)+1}/{self.N+1}")
        self.y_buf.append(float(y_new))
        self.t_buf.append(float(t_new))
        if u_new is not None:
            self.u_buf.append(float(u_new))

        if len(self.y_buf) < 1 or len(self.t_buf) < 1:
            return {"ready": False, "message": "Need at least one (y,t)."}

        M = len(self.y_buf) - 1
        # window arrays (use the most recent M+1 values)
        y_win = np.asarray(list(self.y_buf), dtype=np.float32)[-(M+1):]
        t_win = np.asarray(list(self.t_buf), dtype=np.float32)[-(M+1):]

        y_buf_tf = tf.convert_to_tensor(y_win, dtype=tf.float32)
        t_buf_tf = tf.convert_to_tensor(t_win, dtype=tf.float32)

        # initial guess z0 (positive); convert to unconstrained by inverse-softplus-ish
        z0 = self._initial_guess_M(M).astype(np.float32)
        # map positive z0 -> z0_u so softplus(z0_u) ~ z0
        z0_u = np.log(np.expm1(np.maximum(z0, 1e-12))).astype(np.float32)

        z_u = tf.Variable(z0_u, dtype=tf.float32)

        # objective: 0.5 * ||r||^2
        def value_and_gradients_fn(z_u_flat):
            with tf.GradientTape() as tape:
                tape.watch(z_u_flat)
                r = self._residuals_M_tf(z_u_flat, M, y_buf_tf, t_buf_tf)
                loss = 0.5 * tf.reduce_sum(tf.square(r))
            g = tape.gradient(loss, z_u_flat)
            return loss, g
        self.log(2, f"[SOLVE] M={M}, vars={(M+1)*self.n}, "
            f"t_win={t_win[0]:.2f}->{t_win[-1]:.2f}, "
            f"dt_last={(t_win[-1]-t_win[-2]) if len(t_win)>1 else 0:.2f}")
        self.log(2, f"[SOLVE] z0 min/max {z0.min():.3e}/{z0.max():.3e}")

        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=value_and_gradients_fn,
            initial_position=tf.convert_to_tensor(z_u),
            max_iterations=self.max_nfev,
            tolerance=1e-12
        )


        z_u_opt = results.position
        z_opt   = (tf.nn.softplus(z_u_opt) + tf.cast(self.eps_y, tf.float32)).numpy()
        X_hat   = z_opt.reshape(M + 1, self.n).astype(float)

        self._x_seq_last = X_hat
        self.x_prior = X_hat[-1].copy()

        return {
            "ready": True,
            "success": bool(results.converged.numpy()),
            "status": int(results.num_iterations.numpy()),
            "cost": float(results.objective_value.numpy()),
            "nfev": int(results.num_iterations.numpy()),
            "x_current": X_hat[-1].copy(),
            "X_sequence": X_hat,
        }



    def HPINN_model(self, n_simulation_steps: int):
        cfg_dir = PROJECT_ROOT / "config"
        trained_k_yaml = cfg_dir / "trained_params.yaml"

        t_sim = np.arange(n_simulation_steps, dtype=np.float32) * DT

        model, _, _ = build_model_from_config(
            cfg_dir=cfg_dir,
            trained_k_yaml=trained_k_yaml,
            t_sim=t_sim,
            dt=DT,
            initial_states=np.zeros((1, 8), dtype=np.float32)
        )
        return model


# -----------------------------
# Demo helpers
# -----------------------------
def make_irregular_measurement_times(t0: float, tf: float, mean_step: float, seed: int = 0):
    """
    Generate monotonically increasing irregular timestamps.
    """
    rng = np.random.default_rng(seed)
    times = [t0]
    while times[-1] < tf:
        # Exponential step gives irregular sampling
        dt = rng.exponential(mean_step)
        times.append(times[-1] + dt)
    times = np.array(times, dtype=float)
    times = times[times <= tf]
    return times

def simulate_truth_with_hpinn(mhe: MovingHorizonEstimator, x0: np.ndarray, times: np.ndarray):
    """
    Generate "ground truth" by propagating one interval at a time with the HPINN.

    Uses mhe.f_batch with batch size B=1:
      X0_batch shape: (1, 8)
      time_starts shape: (1,)
      measurement_times shape: (1,)
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 1:
        raise ValueError("times must be a 1D array with at least one element")

    xk = np.asarray(x0, dtype=np.float32).reshape(1, 8)  # (1,8)
    X_true = [xk[0].copy()]

    for k in range(1, len(times)):
        t_prev = float(times[k - 1])
        t_now  = float(times[k])

        X_next = mhe.f_batch(
            X0_batch=xk,                             # (1,8)
            time_starts=np.array([t_prev], float),    # (1,)
            measurement_times=np.array([t_now], float) # (1,)
        )  # (1,8)

        xk = X_next.astype(np.float32)  # keep (1,8)
        X_true.append(xk[0].copy())

    return np.vstack(X_true)  # (K, 8)


def add_relative_noise(y: np.ndarray, rel_sigma: float, seed: int = 1):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, rel_sigma, size=y.shape)
    return y * (1.0 + eps)


def load_batch_data(csv_path: Path, sequence_id: int):
    """
    Load one PFAS batch sequence from CSV.

    Returns:
        t_meas : (K,) float
        Y_meas : (K, 6) float   [C7, C5, C3, C2, CF3, F-]
    """
    df = pd.read_csv(csv_path)

    # Filter one batch
    df = df[df["sequence_id"] == sequence_id].copy()
    if df.empty:
        raise ValueError(f"No data found for sequence_id={sequence_id}")

    # Sort by time just in case
    df = df.sort_values("time (s)")

    t_meas = df["time (s)"].to_numpy(dtype=float)

    Y_meas = df[
        ["C7F15COO-", "C5F11COO-", "C3F7COO-", "C2F5COO-", "CF3COO-", "F-"]
    ].to_numpy(dtype=float)

    return t_meas, Y_meas

# -----------------------------
# Main demo
# -----------------------------
if __name__ == "__main__":
    # We set n=8 in this demo, because your HPINN state is 8D and your h(x)=x[-1].
    n = 8
    print(tf.config.list_physical_devices("GPU"))
    # Choose a sampling_period that covers the largest expected gap between timestamps
    # Here we generate mean ~10s but can have larger; set to 200s to be safe.
    sampling_period = 650

    # Horizon in number of measurements (N)
    horizon = 3

    mhe = MovingHorizonEstimator(
        n=n,
        sampling_period=sampling_period,
        horizon=horizon,
        Q_diag=3.69e-16,
        R_rel=1e-4,
        P0_diag=1.0,
        enforce_nonneg=True,
        use_log_measurement=True,
        max_nfev=200
    )

    # --------------------------------------------------
    # Load REAL PFAS batch data
    # --------------------------------------------------
    DATA_PATH = PROJECT_ROOT / "data" / "Batch_PFAS_data.csv"
    SEQUENCE_ID = 1   # <-- change to 2 to use the second batch

    t_meas, Y_meas = load_batch_data(DATA_PATH, sequence_id=SEQUENCE_ID)

    # We measure fluoride only (matches h(x) = x[-1])
    y_meas = Y_meas[:, -1]   # F-

    print(f"Loaded {len(t_meas)} measurements from sequence {SEQUENCE_ID}")
    print("t range:", t_meas[0], "→", t_meas[-1])
    print("F- range:", y_meas.min(), "→", y_meas.max())

    # Inputs u_k (optional) for each interval
    u_intervals = np.zeros(len(t_meas) - 1, dtype=float)

    x0 = np.zeros(8, dtype=np.float32)

    # Use first measurement row to seed the chemical states
    x0[0] = Y_meas[0, 0]   # C7
    x0[2] = Y_meas[0, 1]   # C5
    x0[4] = Y_meas[0, 2]   # C3
    x0[5] = Y_meas[0, 3]   # C2
    x0[6] = Y_meas[0, 4]   # CF3
    x0[7] = Y_meas[0, 5]   # F-

    x0 = np.maximum(x0, 1e-12)   # positivity floor

    mhe.set_prior(x0)

    # Run the MHE sequentially
    x_est_list = []
    t_est_list = []
    y_hat_list = []
    updates = []

    for k in range(len(t_meas)):
        t_k = float(t_meas[k])
        y_k = float(y_meas[k])

        u_prev = None if k == 0 else float(u_intervals[k-1])

        out = mhe.update(y_new=y_k, t_new=t_k, u_new=u_prev)
        if out.get("ready", False):
            x_hat = out["x_current"]
            x_est_list.append(x_hat)
            t_est_list.append(t_k)
            y_hat_list.append(float(x_hat[-1]))
            updates.append(out)

    
    # --------------------------------------------------
    # Build "data-as-state" array for plotting
    # (8D state, but only 6 measured channels -> fill the rest with NaN)
    # --------------------------------------------------
    t_meas = np.asarray(t_meas, dtype=float)

    X_data = np.full((len(t_meas), 8), np.nan, dtype=float)

    # Your implied HPINN state order (based on how you seeded x0):
    # x[0]=C7, x[2]=C5, x[4]=C3, x[5]=C2, x[6]=CF3, x[7]=F-
    idx_map = [0, 2, 4, 5, 6, 7]  # where the 6 measured species live in the 8D state

    # Y_meas columns are: [C7, C5, C3, C2, CF3, F-]
    X_data[:, idx_map] = Y_meas

    # MHE estimates are only available once ready; align arrays
    if len(x_est_list) > 0:
        t_est = np.asarray(t_est_list, dtype=float)
        X_mhe = np.vstack(x_est_list).astype(float)  # (#solves,8)
    else:
        t_est = np.array([], dtype=float)
        X_mhe = np.zeros((0, 8), dtype=float)

    # --- Model (HPINN) prediction on a dense grid from x0 ---
    # Dense time grid: 1s resolution from 0 to last measurement time
    t0 = float(t_meas[0])
    t_end = float(t_meas[-1])
    t_grid = np.arange(t0, t_end + DT, DT, dtype=float)

    # We need to propagate from x0 at t0 to each t_grid point.
    # Easiest: do it cumulatively (one big rollout) if the grid fits in sampling_period.
    # Here: create a temporary model with enough sim steps for the whole horizon.
    n_steps_grid = int(np.ceil((t_end - t0) / DT)) + 1

    # Build a temporary HPINN model for the whole grid horizon
    cfg_dir = PROJECT_ROOT / "config"
    trained_k_yaml = cfg_dir / "trained_params.yaml"
    t_sim_grid = np.arange(n_steps_grid, dtype=np.float32) * DT

    model_grid, _, _ = build_model_from_config(
        cfg_dir=cfg_dir,
        trained_k_yaml=trained_k_yaml,
        t_sim=t_sim_grid,
        dt=DT,
        initial_states=np.zeros((1, 8), dtype=np.float32)
    )

    dummy_grid = tf.zeros((1, len(t_sim_grid), 1), dtype=tf.float32)
    x0_tf = tf.convert_to_tensor(np.asarray(x0, np.float32).reshape(1, 8), dtype=tf.float32)

    x_series_grid = model_grid([dummy_grid, x0_tf], training=False).numpy()[0]  # (T,8)
    X_model = np.vstack([np.asarray(x0, float), x_series_grid.astype(float)])   # (T+1,8)

    # t for X_model corresponds to t0 + k*DT, k=0..T
    t_model = t0 + np.arange(X_model.shape[0], dtype=float) * DT

    # --- Plot all 8 states in a tiled layout ---
    state_labels = [
    r"$C_{7}F_{15}COO^-$ (x_1)",
    r"$x_2$ (unmeasured)",
    r"$C_{5}F_{11}COO^-$ (x_3)",
    r"$x_4$ (unmeasured)",
    r"$C_{3}F_{7}COO^-$ (x_5)",
    r"$C_{2}F_{5}COO^-$ (x_6)",
    r"$CF_{3}COO^-$ (x_7)",
    r"$F^-$ (x_8)",
    ]


    fig, axes = plt.subplots(8, 1, sharex=True, figsize=(10, 14))
    for i, ax in enumerate(axes):
        # Model prediction (dense)
        ax.plot(t_model, X_model[:, i], "-", linewidth=1.8, label="HPINN model (open-loop)")


        # MHE estimates (only after ready)
        if X_mhe.shape[0] > 0:
            ax.plot(t_est, X_mhe[:, i], "s-", markersize=4, linewidth=1.2, label="MHE estimate")

        # Real measured data points (only where available; NaNs won't plot)
        ax.plot(t_meas, X_data[:, i], "o", markersize=4, label="Data points")

        ax.set_ylabel(state_labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")

    # Put a single legend at top (avoid repeating)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)

    fig.suptitle("Full state trajectory: HPINN model vs MHE vs data", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mhe_tf_seq{SEQUENCE_ID}_N{horizon}_dt{DT:g}_sp{sampling_period}_{stamp}"

    # --------------------------------------------------
    # Save figure to /results
    # --------------------------------------------------
    fig_path_png = results_dir / f"{run_name}.png"
    fig_path_pdf = results_dir / f"{run_name}.pdf"

    fig.savefig(fig_path_png, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path_pdf, bbox_inches="tight")

    print(f"[SAVE] Figure written:\n  {fig_path_png}\n  {fig_path_pdf}", flush=True)


    npz_path = results_dir / f"{run_name}.npz"
    json_path = results_dir / f"{run_name}.json"

    # Save arrays (everything needed to replot)
    np.savez_compressed(
        npz_path,
        # raw measurements
        t_meas=t_meas,
        Y_meas=Y_meas,
        y_meas=y_meas,
        X_data=X_data,

        # MHE results
        t_est=t_est,
        X_mhe=X_mhe,
        y_hat=np.asarray(y_hat_list, dtype=float),

        # HPINN open-loop rollout used for plotting
        t_model=t_model,
        X_model=X_model,

        # initial condition and mapping info
        x0=np.asarray(x0, dtype=float),
        idx_map=np.asarray(idx_map, dtype=int),
    )

    # Save metadata (human-readable)
    meta = {
        "sequence_id": int(SEQUENCE_ID),
        "horizon": int(horizon),
        "DT": float(DT),
        "sampling_period": float(sampling_period),
        "n": int(n),
        "Q_diag": float(3.69e-16),  # or store your actual Q/P/R if you want
        "R_rel": float(1e-4),
        "P0_diag": float(1.0),
        "use_log_measurement": True,
        "enforce_nonneg": True,
        "created": stamp,
        "npz_file": npz_path.name,
    }

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] Wrote:\n  {npz_path}\n  {json_path}", flush=True)