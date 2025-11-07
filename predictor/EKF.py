import numpy as np
from jacobian_rk4 import Jacobian

class ExtendedKalmanFilter:
    def __init__(self, Q, R, x0, P0, k, c_eaq, x_scale, dt=1.0, eps=1e-20,
                 # --- robust knobs (defaults are conservative) ---
                 fading_beta=1.05,          # inflate P ~5%/step to avoid over-confidence
                 q_floor_frac=0.01,         # per-state Q floor = (q_floor_frac * x_scale)^2
                 use_adaptive_R=False,       # innovation-based R tuning
                 R_alpha=0.05,              # EMA rate for R
                 R_floor=0.0,               # minimal R
                 gate_chi2=True,            # reject outliers by NIS
                 gate_prob=0.995):          # chi2 threshold prob for m=1
        """
        Same signature; adds robust defaults to prevent noise amplification.
        """
        # ---- Normalize inputs / shapes (NO API CHANGE) ----
        x0 = np.asarray(x0, float).reshape(-1)             # (n,)
        x_scale = np.asarray(x_scale, float).reshape(-1)   # (n,)
        Q = np.asarray(Q, float)
        R = np.asarray(R, float)
        P0 = np.asarray(P0, float)

        self.n = x0.shape[0]
        assert x_scale.shape == (self.n,), f"x_scale must be length {self.n}, got {x_scale.shape}"

        # Measurement dims (fluoride only, but be robust)
        if R.ndim == 0:
            R = np.array([[float(R)]], dtype=float)
        elif R.ndim == 1:
            R = R.reshape(1, 1)
        self.m = R.shape[0]

        # ---- Store scalars / eps ----
        self.dt = float(dt)
        self.eps = float(eps)

        # Robust knobs
        self.fading_beta = float(fading_beta)
        self.use_adaptive_R = bool(use_adaptive_R)
        self.R_alpha = float(R_alpha)
        self.R_floor = float(R_floor)
        self.gate_chi2 = bool(gate_chi2)
        self.gate_prob = float(gate_prob)

        # ---- Scaling matrices (state only; measurement unscaled) ----
        self.Dx = np.diag(x_scale)
        self.Dx_inv = np.diag(1.0 / x_scale)

        # ---- Jacobians in physical units ----
        self.J = Jacobian(k, c_eaq, dt=self.dt)
        F = np.asarray(self.J.J_reaction, float)      # (n,n)
        H = np.asarray(self.J.J_observation, float)   # (m,n)

        # ---- Transform model/covariances to scaled coordinates ----
        # x' = Dx^{-1} x ; run EKF in x'-space
        from scipy.linalg import expm
        F_disc = expm(F * self.dt)        # discrete-time transition matrix
        self.Fs = self.Dx_inv @ F_disc @ self.Dx
        self.Hs = H @ self.Dx
        self.Qs = self.Dx_inv @ Q @ self.Dx_inv.T
        self.Rs = R.copy()

        # ---- Initialize in scaled coordinates ----
        self.xs = self.Dx_inv @ x0
        self.Ps = self.Dx_inv @ P0 @ self.Dx_inv.T

        # Stabilize numerics
        self.Ps = 0.5 * (self.Ps + self.Ps.T) + self.eps * np.eye(self.n)
        self.Qs = 0.5 * (self.Qs + self.Qs.T) + self.eps * np.eye(self.n)

        # Q floor (scaled coords): keeps weakly observed states corrigible
        q_floor_phys = (float(q_floor_frac) * np.diag(self.Dx)) ** 2  # (n,)
        self.Qs_floor = self.Dx_inv @ np.diag(q_floor_phys) @ self.Dx_inv.T

        # Cache gain
        self.Ks = np.zeros((self.n, self.m))

        # Keep params so set_c_eaq can refresh
        self.k = k
        self.c_eaq = c_eaq

        # Precompute chi2 threshold for m measurements (m=1 here)
        if self.gate_chi2:
            # For m=1, chi2(df=1) ppf at gate_prob
            # 0.99 -> 6.63 ; 0.995 -> 7.88 ; 0.999 -> 10.83
            from math import sqrt
            # closed form for df=1: x = (erfcinv(2p)*sqrt(2))^2, but keep a safe constant if SciPy not available
            try:
                from scipy.stats import chi2
                self._chi2_thr = float(chi2.ppf(self.gate_prob, df=self.m))
            except Exception:
                self._chi2_thr = 7.879  # ≈ chi2.ppf(0.995,1)
        else:
            self._chi2_thr = None

    # Measurement function in scaled space: h'(x') = h(Dx x')   (z is unscaled)
    def h_scaled(self, xs):
        x_phys = self.Dx @ xs
        return np.array([x_phys[7]], dtype=float).reshape(self.m,)

    def predict(self, x):
        """
        We treat provided x as the plant-simulated physical state and
        set the EKF prior mean to that (in scaled space) before covariance propagation.
        For covariance, since the mean came from the plant, we use identity transition.
        """
        x = np.asarray(x, float).reshape(-1)
        assert x.shape == (self.n,), f"predict() expected state shape {(self.n,)}, got {x.shape}"

        # Prior mean from plant (scaled) + non-negativity for concentrations
        self.xs = np.maximum(self.Dx_inv @ x, 0.0)

        # ---- STABLE COVARIANCE PREDICTION ----
        # Use identity transition because the mean came from the plant integrator
        Ps_pred = self.Ps + self.Qs

        # Add per-state floor to keep weakly observed states corrigible
        Ps_pred = Ps_pred + self.Qs_floor

        # Mild fading (optional). You can set =1.0 if you want none.
        self.Ps = self.fading_beta * Ps_pred  # e.g., 1.05

        # Symmetrize + jitter
        self.Ps = 0.5 * (self.Ps + self.Ps.T)
        self.Ps += self.eps * np.eye(self.n)


    def update(self, z):
        """
        simulate() passes a scalar fluoride measurement. Accept scalar or (1,) or (1,1).
        Returns the physical state estimate (n,).
        """
        z_arr = np.asarray(z, float)
        if z_arr.ndim == 0:
            zs = np.array([float(z_arr)], dtype=float)
        elif z_arr.ndim == 1:
            zs = z_arr.reshape(-1)
        else:
            zs = z_arr.reshape(-1)
        assert zs.shape == (self.m,), f"measurement z must have shape {(self.m,)}, got {zs.shape}"
        print(f"[update-pre] id={id(self)}  Rs(before)={self.Rs}  use_adaptive_R={self.use_adaptive_R}")
        # Innovation
        h = self.h_scaled(self.xs)
        nu = zs - h                    # (m,)
        S  = self.Hs @ self.Ps @ self.Hs.T + self.Rs
        S += self.eps * np.eye(self.m)

        # Optional chi-square gating (skip update on outlier)
        if self.gate_chi2:
            nis = float(nu.T @ np.linalg.inv(S) @ nu)
            if nis > self._chi2_thr:
                # Return prior (physical units)
                return self.Dx @ self.xs

        # Gain
        self.Ks = self.Ps @ self.Hs.T @ np.linalg.inv(S)

        # State update (scaled)
        self.xs = self.xs + self.Ks @ nu
        self.xs = np.maximum(self.xs, 0.0)

        # Joseph form covariance update (scaled)
        I = np.eye(self.n)
        IKH = I - self.Ks @ self.Hs
        self.Ps = IKH @ self.Ps @ IKH.T + self.Ks @ self.Rs @ self.Ks.T

        # Symmetrize + jitter
        self.Ps = 0.5 * (self.Ps + self.Ps.T)
        self.Ps += self.eps * np.eye(self.n)

        # Adaptive R via covariance matching (kept tiny & safe)
        #if self.use_adaptive_R:
        #    HPHT = self.Hs @ self.Ps @ self.Hs.T
        #    R_hat = np.outer(nu, nu) - HPHT
        #    # keep PSD-ish and floor
        #    R_hat = 0.5 * (R_hat + R_hat.T)
        #    self.Rs = (1.0 - self.R_alpha) * self.Rs + self.R_alpha * R_hat
        #    self.Rs = np.maximum(self.Rs, self.R_floor)
        #    self.Rs = 0.5 * (self.Rs + self.Rs.T) + self.eps * np.eye(self.m)

        print("new R:", self.Rs)
        # Return physical units
        
        return self.Dx @ self.xs

    def get_state(self):
        """Return (x, P) in physical units for logging/plotting."""
        x = self.Dx @ self.xs
        P = self.Dx @ self.Ps @ self.Dx.T
        return x, P

    def set_c_eaq(self, c_eaq):
        """simulate() calls this; refresh Jacobians and keep scaling consistent."""
        self.c_eaq = c_eaq
        self.J.update(c_eaq=c_eaq)
        F = np.asarray(self.J.J_reaction, float)
        H = np.asarray(self.J.J_observation, float)
        self.Fs = self.Dx_inv @ F @ self.Dx
        self.Hs = H @ self.Dx

    def set_R(self, R_phys: np.ndarray):
        R = np.asarray(R_phys, float).reshape(self.m, self.m)
        # symmetry + tiny jitter
        R = 0.5*(R + R.T) + self.eps*np.eye(self.m)
        # optional floor: comment out if you don't want it
        R = np.maximum(R, self.R_floor)

        self.Rs = R
        self.R = R


    # Back-compat convenience (physical units)
    @property
    def x(self):  # mean
        return (self.Dx @ self.xs).reshape(-1)
    @property
    def P(self):  # covariance
        return self.Dx @ self.Ps @ self.Dx.T
    @property
    def K(self):  # gain (scaled coords)
        return self.Ks


"""
import numpy as np
from scipy.linalg import expm
from jacobian_rk4 import Jacobian

class ExtendedKalmanFilter:
    def __init__(self, Q, R, x0, P0, k, c_eaq,x_scale, dt=1.0):

        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

        self.J = Jacobian(k, c_eaq, dt=dt)
        self.F = self.J.J_reaction
        self.H = self.J.J_observation

        self.k = k
        self.dt = float(dt)
        self.eps = 1e-9
        self.x_scale = x_scale
        

    def predict(self, x):
        F = self.F

        # Normalize state
        self.x = x/self.x_scale

        # Enforce simple non-negativity for concentrations (keeps filter from going unphysical).
        # If you have states that may be negative by definition, exclude them from this clip.
        self.x = np.maximum(self.x, 0.0)

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

        # Symmetrize + jitter to keep P PSD
        self.P = 0.5 * (self.P + self.P.T)
        self.P += self.eps * np.eye(self.P.shape[0])

        

    def update(self, z):
        H = self.H
        # Normalize measurement
        z = z
        y = z - self.h(self.x)                # innovation
        S = H @ self.P @ H.T + self.R
        S = S + self.eps * np.eye(S.shape[0]) # jitter

        self.K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + self.K @ y

        # Optional: keep concentrations non-negative after update as well
        self.x = np.maximum(self.x, 0.0)

        I = np.eye(self.P.shape[0])
        self.P = (I - self.K @ H) @ self.P @ (I - self.K @ H).T + self.K @ self.R @ self.K.T

        # Symmetrize + jitter
        self.P = 0.5 * (self.P + self.P.T)
        self.P += self.eps * np.eye(self.P.shape[0])
        return self.x*self.x_scale

    def h(self, x):
        Observation function h(x) that maps state to measurement.
        In this case, we assume we measure the fluoride concentration.
        return np.array([x[7]]) # fluoride concentration is the 8th state (index 7)
    
    def set_c_eaq(self, c_eaq):
        self.J.update(c_eaq=c_eaq)
        self.F = self.J.J_reaction 
"""
