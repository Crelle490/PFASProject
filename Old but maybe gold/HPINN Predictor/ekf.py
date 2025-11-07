# ekf.py  — minimal, modular EKF

import numpy as np

# ekf_cov.py (or add to ekf.py)
import numpy as np, yaml
from pathlib import Path

# Load covariance matrices from config/covariance_params.yaml
# Returns Q, R, P0 as numpy arrays.
def load_covariances(cfg_dir, n_states, m_meas):
    """
    Load Q, R, P0 from config/covariance_params.yaml and slice to sizes.
    Returns:
      Q  : (n_states, n_states)
      R  : (m_meas,  m_meas)
      P0 : (n_states, n_states)
    """
    d = yaml.safe_load(open(Path(cfg_dir) / "covariance_params.yaml", "r"))
    Q  = np.array(d["process_noise_covariance"],    dtype=np.float32)[:n_states, :n_states]
    R  = np.array(d["measurement_noise_covariance"],dtype=np.float32)[:m_meas,  :m_meas]
    P0 = np.array(d["initial_error_covariance"],    dtype=np.float32)[:n_states, :n_states]
    return Q, R, P0

# Take one EKF step
def ekf_step(x, P, z, *, predict_fn, measure_fn, A_fn, H_fn, Q, R):
    """
    One EKF step (modular).
    x: (n,)   current state estimate (all concentrations)
    P: (n,n)  current covariance
    z: (m,)   current measurement (e.g., fluoride)
    predict_fn: f(x) -> x_pred
    measure_fn: h(x_pred) -> z_pred
    A_fn: A(x) = ∂f/∂x at x         -> (n,n)
    H_fn: H(x_pred) = ∂h/∂x at x_pred -> (m,n)
    Q: (n,n) process noise
    R: (m,m) measurement noise
    returns: (x_new, P_new)
    """
    x_pred = predict_fn(x) # Kinetic model estimation
    A = A_fn(x)
    P_pred = A @ P @ A.T + Q

    z_pred = measure_fn(x_pred) # Measurement prediction

    H = H_fn(x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S) # Kalman gain - model vs measurement

    y = z - z_pred
    x_new = x_pred + K @ y # Fused state estimate
    P_new = (np.eye(P.shape[0]) - K @ H) @ P_pred # Updated covariance


    return x_new.astype(np.float32), P_new.astype(np.float32)

# Preare EKF parameters
def ekf_prepare(x0, P0, Q, R, dtype=np.float32):
    """Return x, P, Q, R as same dtype (no other logic)."""
    x = np.asarray(x0, dtype)
    P = np.asarray(P0, dtype)
    Q = np.asarray(Q,  dtype)
    R = np.asarray(R,  dtype)
    return x, P, Q, R


# Complete EKF run over a series of measurements
def ekf_run(z_series, x, P, *, predict_fn, measure_fn, A_fn, H_fn, Q, R, clamp_nonneg=True):
    Z = np.atleast_2d(np.asarray(z_series, np.float32))
    if Z.shape[0] == 1 and Z.shape[1] > 1:
        Z = Z.T
    X = np.zeros((Z.shape[0] + 1, x.size), x.dtype)
    X[0] = x
    for k in range(Z.shape[0]):
        x, P = ekf_step(x, P, Z[k], predict_fn=predict_fn, measure_fn=measure_fn,
                        A_fn=A_fn, H_fn=H_fn, Q=Q, R=R)
        if clamp_nonneg:
            x = np.maximum(x, 0.0)
        X[k+1] = x
    return X, P

