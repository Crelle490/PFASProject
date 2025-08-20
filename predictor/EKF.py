import numpy as np
from scipy.linalg import expm
from .jacobian_rk4 import Jacobian

class ExtendedKalmanFilter:
    def __init__(self, Q, R, x0, P0, k, c_eaq, dt=1.0):
        """
        f: Takes predictions from HPINN model
        h: Observation function h(x)
        F_jacobian: function that computes Jacobian of f w.r.t x (current implementation assuems constant c_eaq into account, hence the jacobian is constatn)
        H_jacobian: function that computes Jacobian of h w.r.t x (current implementation assuems constant c_eaq into account, hence the jacobian is constatn)
        Q: process noise covariance
        R: measurement noise covariance
        x0: initial state estimate
        P0: initial error covariance
        k: reaction rate constants
        """
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
        

    def predict(self, x, u):
        F = self.F

        # (Optional) monitor spectral radius for sanity
        rho = np.max(np.abs(np.linalg.eigvals(F)))
        print("rho(F) =", float(rho))

        # Propagate state
        self.x = x

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

    def h(self, x):
        """
        Observation function h(x) that maps state to measurement.
        In this case, we assume we measure the fluoride concentration.
        """
        return np.array([x[7]]) # fluoride concentration is the 8th state (index 7)

