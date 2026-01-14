import sys
from pathlib import Path

import numpy as np

# Allow running as a script when project root isn't on PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.moving_horizon_estimator import HPINNMovingHorizonEstimator


def main():
    # Configure MHE
    dt = 1.0
    horizon_steps = 20
    meas_idx = (7,)  # fluoride
    mhe = HPINNMovingHorizonEstimator(
        dt=dt,
        horizon_steps=horizon_steps,
        measurement_indices=meas_idx,
        max_iters=200,
        learning_rate=1e-2,
    )

    # Create a synthetic measurement stream from the HPINN model.
    # This keeps the example self-contained and shows how the MHE fits to data.
    t_total = horizon_steps
    x0 = np.copy(mhe._x0_guess)  # use MHE initial guess as ground truth
    y_true = mhe.simulate(x0, t_total)
    z_true = y_true[:, meas_idx[0]]

    rng = np.random.default_rng(0)
    meas_noise_std = 1e-4
    z_meas = z_true + rng.normal(0.0, meas_noise_std, size=z_true.shape).astype(np.float32)

    print("Step | z (meas) | F- (est)")
    print("---------------------------")
    for k in range(t_total):
        state, _traj = mhe.step([float(z_meas[k])])
        print(f"{k:4d} | {z_meas[k]:.6f} | {float(state[7]):.6f}")


if __name__ == "__main__":
    main()
