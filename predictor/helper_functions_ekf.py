# predictor/helper_functions_ekf.py
import numpy as np

try:
    from PFASProject.predictor.EKF import ExtendedKalmanFilter
except Exception:
    from EKF import ExtendedKalmanFilter

# optional builder
try:
    from PFASProject.predictor.covariance_builder import make_covariances_for_fluoride_only
except Exception:
    try:
        from covariance_builder import make_covariances_for_fluoride_only
    except Exception:
        make_covariances_for_fluoride_only = None


def _extract_covariances(cov_params, x_scale, default_meas_std=2e-9,
                         alpha=0.02, p0_frac=0.5, rho=0.0):
    """
    Accepts either:
      - {'Q', 'R', 'P0'}  (numpy arrays)
      - {'process_noise_covariance', 'measurement_noise_covariance', 'initial_error_covariance'}
    or, if missing, builds from x_scale using make_covariances_for_fluoride_only.
    Returns (Q, R, P0) in PHYSICAL units.
    """
    if cov_params is None:
        cov_params = {}

    # Try compact keys first
    Q = cov_params.get("Q", None)
    R = cov_params.get("R", None)
    P0 = cov_params.get("P0", None)

    # Then YAML-style keys
    if Q is None and "process_noise_covariance" in cov_params:
        Q = np.array(cov_params["process_noise_covariance"], dtype=float)
    if R is None and "measurement_noise_covariance" in cov_params:
        R = np.array(cov_params["measurement_noise_covariance"], dtype=float)
    if P0 is None and "initial_error_covariance" in cov_params:
        P0 = np.array(cov_params["initial_error_covariance"], dtype=float)

    # If still missing any, build from x_scale
    if (Q is None) or (R is None) or (P0 is None):
        if make_covariances_for_fluoride_only is None:
            missing = [k for k, v in (("Q", Q), ("R", R), ("P0", P0)) if v is None]
            raise KeyError(
                f"Missing covariances {missing}. "
                f"Provide them in cov_params or add covariance_builder.py."
            )
        Q, R, P0 = make_covariances_for_fluoride_only(
            x_scale=np.asarray(x_scale, float).reshape(-1),
            meas_std=float(default_meas_std),
            alpha=float(alpha),
            p0_frac=float(p0_frac),
            rho=float(rho),
        )

    # Ensure correct shapes
    Q = np.asarray(Q, float)
    P0 = np.asarray(P0, float)
    R = np.asarray(R, float)
    if R.ndim == 0:
        R = R.reshape(1, 1)
    elif R.ndim == 1:
        R = R.reshape(1, 1)

    return Q, R, P0


def init_EKF(
    x0,
    dt_sim,
    c_eaq,
    k,
    cov_params,
    x_scale,
    *,
    use_adaptive_R=True,
    R_alpha=0.05,
    R_floor=0.0,
    fading_beta=1.0,
    Q_floor_diag=None,
    gate_chi2=True,
    gate_prob=0.995,
):
    """Initialize EKF; accepts both YAML-style and compact covariance keys."""
    x0 = np.asarray(x0, float).reshape(-1)
    assert x0.size == 8, f"x0 must have 8 elements after reshape, got {x0.size}"
    x_scale = np.asarray(x_scale, float).reshape(-1)
    assert x_scale.size == 8, f"x_scale must have length 8, got {x_scale.size}"

    Q, R, P0 = _extract_covariances(cov_params, x_scale)

    ekf = ExtendedKalmanFilter(
        Q=Q,
        R=R,
        x0=x0,
        P0=P0,
        k=k,
        c_eaq=c_eaq,
        x_scale=x_scale,
        dt=float(dt_sim),
    )

    # Optional robust knobs
    if hasattr(ekf, "use_adaptive_R"): ekf.use_adaptive_R = bool(use_adaptive_R)
    if hasattr(ekf, "R_alpha"):        ekf.R_alpha = float(R_alpha)
    if hasattr(ekf, "R_floor"):        ekf.R_floor = float(R_floor)
    if hasattr(ekf, "fading_beta"):    ekf.fading_beta = float(fading_beta)
    if hasattr(ekf, "gate_chi2"):      ekf.gate_chi2 = bool(gate_chi2)
    if hasattr(ekf, "gate_prob"):      ekf.gate_prob = float(gate_prob)

    if hasattr(ekf, "Dx_inv") and Q_floor_diag is not None and hasattr(ekf, "Qs_floor"):
        qf = np.asarray(Q_floor_diag, float).reshape(-1)
        assert qf.size == 8, "Q_floor_diag must have length 8"
        ekf.Qs_floor = ekf.Dx_inv @ np.diag(qf) @ ekf.Dx_inv.T

    # Backward-compatible debug print (relies on ekf.x / ekf.P properties)
    try:
        print("EKF init → x:", ekf.x.shape, "P:", ekf.P.shape)
    except Exception:
        pass

    return ekf
