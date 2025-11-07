# predictor/covariance_builder.py
import numpy as np

import numpy as np

def make_covariances_for_fluoride_only(
    x_scale: np.ndarray,
    meas_std: float,
    p0_frac: float = 0.5,     # initial 1σ ~ 50% of typical state scale
    rho: float = 0.0          # nearest-neighbor process correlation (0..0.3 typical)
):
    """
    Returns Q (8x8), R (1x1), P0 (8x8) in *physical units*.
    Assumes only fluoride (state index 7) is measured.

    x_scale: (8,) per-state typical magnitudes (your printed 'z_scale' vector)
    meas_std: fluoride sensor std-dev (same units as fluoride state), e.g. 2e-9
    p0_frac: fractional initial uncertainty   (P0_ii = (p0_frac*x_scale_i)^2)
    rho: optional nearest-neighbor correlation on Q (symmetric)

    α_i grows with state index so later states are modeled as less certain:
        α = [0.01, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.025]
    """

    x_scale = np.asarray(x_scale, float).reshape(-1)
    assert x_scale.shape[0] == 8, f"Expected x_scale of length 8, got {x_scale.shape}"

    # --- Fractional process noise per state (later states more uncertain)
    alpha_vec = np.linspace(0.01, 0.025, 8)  # from 1% to 2.5% per control step

    # --- Process noise covariance
    q_diag = (alpha_vec * x_scale) ** 2
    Q = np.diag(q_diag)

    # Optional nearest-neighbor correlation (small and symmetric)
    if rho and rho != 0.0:
        for i in range(7):
            q_ij = rho * np.sqrt(q_diag[i] * q_diag[i + 1])
            Q[i, i + 1] = Q[i + 1, i] = q_ij
    Q = 0.5 * (Q + Q.T)  # enforce symmetry

    # --- Measurement noise (fluoride only)
    R = np.array([[float(meas_std) ** 2]], dtype=float)

    # --- Initial covariance: broad but finite belief
    P0 = np.diag((p0_frac * x_scale) ** 2)

    # Report useful diagnostic
    ratio = np.diag(Q)[-1] / R[0, 0]

    return Q, R, P0


import numpy as np

def compute_dynamic_R_from_measurement(
    measurement: float,
    rel_accuracy: float = 0.02,    # ±2% datasheet accuracy → 1σ = 2% of reading
    min_std: float = 1e-7,         # noise floor (same units as measurement)
    units: str = "M"               # "M" (mol/L) or "mg/L"
) -> np.ndarray:
    """
    Compute measurement noise covariance R for the fluoride sensor
    based on the current measurement value.

    Parameters
    ----------
    measurement : float
        Current fluoride measurement (in same units used by EKF).
    rel_accuracy : float, optional
        Relative 1σ accuracy (e.g. 0.02 = ±2% of reading).
    min_std : float, optional
        Minimum standard deviation (noise floor) to avoid R=0 near zero.
    units : str, optional
        "M" or "mg/L" — used only for clarity/documentation.

    Returns
    -------
    R : np.ndarray, shape (1,1)
        Measurement noise covariance matrix for EKF update.
    """

    # sanitize input
    measurement = abs(float(measurement))

    # compute 1σ standard deviation
    sigma = rel_accuracy * measurement
    sigma = max(sigma, min_std)

    # measurement covariance matrix
    R = np.array([[sigma ** 2]], dtype=float)

    return R
