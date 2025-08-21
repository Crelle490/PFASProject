import sympy as sp
import numpy as np
"""
It may be nessecary to check for observability after calculating the local linearization point
"""
class Jacobian:
    def __init__(self, k, c_eaq,dt=1.0):
        self.x = [0]*8
        self.k = k
        self.c_eaq = c_eaq
        self.dt = dt

        # Call method to compute Jacobian function on init
        self.continuous_A_from_params()
        self.rk4_transition_jacobian()
        self.jacobian_observation()

    def continuous_A_from_params(self):
        # k_vals: iterable [k1..k7], c_eaq: float
        k = np.asarray(self.k, dtype=float)
        c = float(self.c_eaq)
        A = np.zeros((8,8), dtype=float)

        A[0,0] = -k[0]*c
        A[1,0] =  k[0]*c;  A[1,1] = -k[1]*c
        A[2,1] =  k[1]*c;  A[2,2] = -k[2]*c
        A[3,2] =  k[2]*c;  A[3,3] = -k[3]*c
        A[4,3] =  k[3]*c;  A[4,4] = -k[4]*c
        A[5,4] =  k[4]*c;  A[5,5] = -k[5]*c
        A[6,5] =  k[5]*c;  A[6,6] = -k[6]*c

        A[7,0] = 2*k[0]*c
        A[7,1] = 2*k[1]*c
        A[7,2] = 2*k[2]*c
        A[7,3] = 2*k[3]*c
        A[7,4] = 2*k[4]*c
        A[7,5] = 2*k[5]*c
        A[7,6] = 2*k[6]*c
        # A[7,7] stays 0
        self.A = A

    def rk4_transition_jacobian(self):
        Z  = self.dt * self.A
        Z2 = Z @ Z
        Z3 = Z2 @ Z
        Z4 = Z3 @ Z
        self.J_reaction = np.eye(self.A.shape[0]) + Z + 0.5*Z2 + (1.0/6.0)*Z3 + (1.0/24.0)*Z4

    def jacobian_observation(self):
        H = np.zeros((1, len(self.x)))
        H[0, 7] = 1  # fluoride is the 8th state (index 7), since the electron state is not insluced in the state matrix
        self.J_observation = H
    
    def update(self, k=None, c_eaq=None, dt=None):
        # Updates the Jacobian based on changes in the parameters. Primarily used for updates in eqa
        if k is not None:      self.k = k
        if c_eaq is not None:  self.c_eaq = c_eaq
        if dt is not None:     self.dt = dt
        self.continuous_A_from_params()
        self.rk4_transition_jacobian()
