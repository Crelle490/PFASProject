import numpy as np
import sympy as sp
from kinetic_model import f,h

class Jacobian:
    def __init__(self, k):
        self.k = k
        self.x_dim = 9
        self.jacobian_reaction_calculate()

    def jacobian_reaction_calculate(self):
        x_syms = sp.symbols('x1:10')  # x1 to x9
        k_syms = sp.symbols('k1:8')   # k1 to k7

        r38 = k_syms[0] * x_syms[0] * x_syms[1]
        r39 = k_syms[1] * x_syms[0] * x_syms[2]
        r40 = k_syms[2] * x_syms[0] * x_syms[4]
        r41 = k_syms[3] * x_syms[0] * x_syms[5]
        r42 = k_syms[4] * x_syms[0] * x_syms[6]
        r43 = k_syms[5] * x_syms[0] * x_syms[7]
        r44 = k_syms[6] * x_syms[0] * x_syms[8]

        dx = [0]*9
        dx[0] = -r38 - r39 - r40 - r41 - r42 - r43 - r44
        dx[1] = -r38
        dx[2] = -r39
        dx[3] = 2 * (r38 + r39 + r40 + r41 + r42 + r43 + r44)
        dx[4] = r39 - r40
        dx[5] = r40 - r41
        dx[6] = r41 - r42
        dx[7] = r42 - r43
        dx[8] = r43 - r44

        J = sp.Matrix(dx).jacobian(x_syms)
        self._x_syms = x_syms
        self._k_syms = k_syms
        self.J_reaction_func = sp.lambdify((x_syms, k_syms), J, modules='numpy')

    def jacobian_reaction(self, x_point):
        return self.J_reaction_func(x_point, self.k)

    def jacobian_observation(self):
        H = np.zeros((1, self.x_dim))
        H[0, 3] = 1  # fluoride concentration at index 3
        return H



class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0, k):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.k = k
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian

    def predict(self, u):
        F = self.F_jacobian(self.x)
        self.x = self.f(self.x, u, self.k)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = self.H_jacobian()
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

# Usage example
k = [0.1, 0.2, 0.1, 0.05, 0.07, 0.03, 0.02]  # example rate constants
x0 = np.zeros(9)
P0 = np.eye(9) * 0.1
Q = np.eye(9) * 0.01
R = np.eye(1) * 0.1

jacobian = Jacobian(k)

ekf = ExtendedKalmanFilter(
    f=f,
    h=h,
    F_jacobian=jacobian.jacobian_reaction,
    H_jacobian=jacobian.jacobian_observation,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    k=k
)

u = 0.05  # some input affecting first state
z = np.array([0.2])  # measurement of fluoride concentration

ekf.predict(u)
ekf.update(z)

print("Updated state estimate:", ekf.x)
