import contextlib
import io
import os
import sys
import csv
from pathlib import Path

# Force matplotlib cache to a writable spot inside the project to avoid permission issues.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".mplcache"))

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Allow running as a script when project root isn't on PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.helper_functions_ekf import init_EKF

try:
    from predictor.moving_horizon_estimator import HPINNMovingHorizonEstimator
except ModuleNotFoundError as exc:
    if exc.name == "tensorflow":
        sys.exit("TensorFlow is required for the MHE. Install tensorflow-cpu (or tensorflow) and retry.")
    raise


REACTION_RATE_SO3_EAQ = 1.5e6
REACTION_RATE_CL_EAQ = 1.0e6
BETA_J = 2.57e4


def load_time_series(data_path: Path, sim_time: float):
    time_raw, fluoride_raw = [], []
    pf_series = {
        "C7F15COO-": [],
        "C5F11COO-": [],
        "C3F7COO-": [],
        "C2F5COO-": [],
        "CF3COO-": [],
    }

    with open(data_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        first_seq = None
        for row in reader:
            seq_id = int(row["sequence_id"])
            if first_seq is None:
                first_seq = seq_id
            if seq_id != first_seq:
                continue

            t = float(row["time (s)"])
            if t > sim_time:
                continue

            time_raw.append(t)
            fluoride_raw.append(float(row["F-"]))
            for col in pf_series:
                pf_series[col].append(float(row[col]))

    if not time_raw:
        raise ValueError(f"No data found in {data_path} up to {sim_time}s")

    time_raw = np.array(time_raw, dtype=float)
    fluoride_raw = np.array(fluoride_raw, dtype=float)
    for col in pf_series:
        pf_series[col] = np.array(pf_series[col], dtype=float)

    pfas_init = float(pf_series["C7F15COO-"][0])
    return time_raw, fluoride_raw, pfas_init, pf_series


def build_x_scale(pf_series: dict, fluoride_series: np.ndarray, pfas_init: float):
    scales = np.full(8, max(pfas_init, 1e-12), dtype=float)
    pf_cols = ["C7F15COO-", "C5F11COO-", "C3F7COO-", "C2F5COO-", "CF3COO-"]

    for idx, col in enumerate(pf_cols):
        if col in pf_series:
            scales[idx] = max(scales[idx], float(np.max(pf_series[col])))

    # Use the last available PFAS scale for unmeasured intermediate species.
    scales[5] = scales[6] = scales[4]
    scales[7] = max(float(np.max(fluoride_series)), 1e-12)
    return scales


def override_initial_pfas(mhe: HPINNMovingHorizonEstimator, c_pfas_init: float):
    mhe.rk_cell.initial_state[0, 0] = np.float32(c_pfas_init)
    mhe._x0_guess = np.asarray([[c_pfas_init] + [0.0] * 7], dtype=np.float32)
    mhe.last_state = mhe._x0_guess[0].copy()
    mhe._window = []


def warm_start_mhe_with_measurement(mhe: HPINNMovingHorizonEstimator, z0: float):
    """Initialize MHE guess/prior using the first fluoride measurement."""
    z0 = float(z0)
    mhe.last_state[7] = z0
    mhe._x0_guess = np.asarray([mhe.last_state], dtype=np.float32)
    mhe._window = []


def compute_c_eaq(rk_cell, k1: float, c_pfas_init: float):
    numerator = rk_cell.generation_of_eaq()
    denominator = (
        k1 * c_pfas_init
        + BETA_J
        + REACTION_RATE_SO3_EAQ * rk_cell.c_so3
        + REACTION_RATE_CL_EAQ * rk_cell.c_cl
    )
    return float(numerator / denominator)


def ekf_update_quiet(ekf, measurement: float):
    with contextlib.redirect_stdout(io.StringIO()):
        return ekf.update(measurement)


def main():
    sim_time = 2000.0
    cfg_dir = PROJECT_ROOT / "config"
    data_path = PROJECT_ROOT / "data" / "Batch_PFAS_data.csv"

    t_raw, fluoride_meas, pfas_init, pf_series = load_time_series(data_path, sim_time)

    cov_params = yaml.safe_load(open(cfg_dir / "covariance_params.yaml", "r"))
    trained_params = yaml.safe_load(open(cfg_dir / "trained_params.yaml", "r"))

    k_vals = np.array([trained_params[f"k{i}"] for i in range(1, 8)], dtype=np.float32)
    # Use the median sample spacing for dt, but cap to keep Jacobian expm stable.
    if len(t_raw) > 1:
        dt_raw = float(np.median(np.diff(t_raw)))
    else:
        dt_raw = 1.0
    dt = min(dt_raw, 1.0)

    # Keep MHE horizon roughly spanning one measurement gap (with a small buffer).
    horizon_steps = 500

    mhe = HPINNMovingHorizonEstimator(
        dt=dt,
        horizon_steps=horizon_steps,
        measurement_indices=(7,),
        max_iters=500,
        learning_rate=1e-3,
    )
    override_initial_pfas(mhe, pfas_init)
    warm_start_mhe_with_measurement(mhe, fluoride_meas[0])

    c_eaq = compute_c_eaq(mhe.rk_cell, float(k_vals[0]), pfas_init)

    x_scale = build_x_scale(pf_series, fluoride_meas, pfas_init)
    x0 = np.zeros(8, dtype=np.float32)
    x0[0] = np.float32(pfas_init)
    dt = 1.0
    
    ekf = init_EKF(
        x0=x0,
        dt_sim=dt,
        c_eaq=c_eaq,
        k=k_vals,
        cov_params=cov_params,
        x_scale=x_scale,
        use_adaptive_R=False,
        fading_beta=1.02,
        gate_chi2=True,
    )

    ekf_states = np.zeros((len(t_raw), 8), dtype=float)
    mhe_states = np.zeros((len(t_raw), 8), dtype=float)

    plant_state = x0.copy()
    prev_t = t_raw[0]

    for idx, z in enumerate(fluoride_meas):
        if idx == 0:
            dt_steps = 1
        else:
            dt_gap = float(t_raw[idx] - prev_t)
            dt_steps = int(max(1, round(dt_gap / dt)))
        prev_t = t_raw[idx]

        # Propagate plant and EKF between measurements
        for _ in range(dt_steps):
            ekf.predict(plant_state)
            plant_state = mhe._rollout_tf(plant_state.reshape(1, -1), 1).numpy()[0].astype(np.float32)

        # Update with measurement
        ekf_states[idx] = ekf_update_quiet(ekf, float(z))

        # Advance MHE prior over the gap, then update window with measurement
        mhe.advance_prior(dt_steps)
        mhe_state, _traj = mhe.step([float(z)])
        mhe_states[idx] = np.maximum(mhe_state, 0.0)

        if idx % 10 == 0 and idx > 0:
            print(f"Processed {idx+1} / {len(fluoride_meas)} measurements")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "ekf_mhe_fluoride_comparison.png"
    csv_path = results_dir / "ekf_mhe_comparison.csv"
    npz_path = results_dir / "ekf_mhe_comparison.npz"

    plt.figure(figsize=(10, 5))
    plt.plot(t_raw, fluoride_meas, label="Measurement (F-)", color="black", linewidth=1.5)
    plt.plot(t_raw, ekf_states[:, 7], label="EKF estimate (F-)", color="tab:blue")
    plt.plot(t_raw, mhe_states[:, 7], label="MHE estimate (F-)", color="tab:orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Fluoride concentration")
    plt.title("EKF vs MHE fluoride estimates (2000 s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)

    # Save numerical results (fluoride only to CSV, full states to NPZ).
    stacked = np.column_stack([t_raw, fluoride_meas, ekf_states[:, 7], mhe_states[:, 7]])
    header = "time_s,meas_F,ekf_F,mhe_F"
    np.savetxt(csv_path, stacked, delimiter=",", header=header, comments="")
    np.savez(npz_path, time=t_raw, meas_F=fluoride_meas, ekf=ekf_states, mhe=mhe_states)

    print(f"Saved comparison plot to {out_path}")
    print(f"Saved fluoride CSV to {csv_path}")
    print(f"Saved full arrays to {npz_path}")
    print(f"Final EKF state: {ekf_states[-1]}")
    print(f"Final MHE state: {mhe_states[-1]}")


if __name__ == "__main__":
    main()
