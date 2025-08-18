import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0,k):
        """
        f: function for state transition f(x, u)
        h: function for observation h(x)
        F_jacobian: function that computes Jacobian of f w.r.t x
        H_jacobian: function that computes Jacobian of h w.r.t x
        Q: process noise covariance
        R: measurement noise covariance
        x0: initial state estimate
        P0: initial error covariance
        k: reaction rate constants
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian

        self.k = k

        

    def predict(self, u):
        F = self.F_jacobian(self.x)
        self.x = self.f(self.x, u, self.k) # Model estimation
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z=z # measurement estimation
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)  # Innovation
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain - model vs measurement
        self.x = self.x + K @ y # fused state estimate
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P # Updated covariance
