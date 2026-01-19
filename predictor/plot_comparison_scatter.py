import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Avoid permission issues with matplotlib cache.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".mplcache"))


def load_results(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data


def main():
    csv_path = PROJECT_ROOT / "results" / "ekf_mhe_comparison.csv"
    out_scatter = PROJECT_ROOT / "results" / "ekf_mhe_scatter.png"
    out_error = PROJECT_ROOT / "results" / "ekf_mhe_errors.png"

    data = load_results(csv_path)
    t = data["time_s"]
    meas = data["meas_F"]
    ekf = data["ekf_F"]
    mhe = data["mhe_F"]

    # Scatter of measurements and estimates
    plt.figure(figsize=(8, 5))
    plt.scatter(t, meas, s=35, color="black", alpha=0.8, marker="o", label="Measurement (F-)")
    plt.scatter(t, ekf, s=30, color="tab:blue", alpha=0.7, marker="s", label="EKF estimate")
    plt.scatter(t, mhe, s=30, color="tab:orange", alpha=0.7, marker="^", label="MHE estimate")
    plt.xlabel("Time (s)")
    plt.ylabel("Fluoride concentration")
    plt.title("EKF vs MHE vs Measurements (scatter)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=300)
    print(f"Saved scatter plot to {out_scatter}")

    # Error plot (estimate minus measurement)
    ekf_err = ekf - meas
    mhe_err = mhe - meas

    plt.figure(figsize=(8, 4))
    plt.plot(t, ekf_err, label="EKF - meas", color="tab:blue")
    plt.plot(t, mhe_err, label="MHE - meas", color="tab:orange")
    plt.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (concentration)")
    plt.title("Estimation error relative to measurement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_error, dpi=300)
    print(f"Saved error plot to {out_error}")


if __name__ == "__main__":
    main()
