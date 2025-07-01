import sympy as sp
import numpy as np
"""
It may be nessecary to check for observability after calculating the local linearization point
"""
class Jacobian:
    def __init__(self, k):
        self.x = [0]*9
        self.k = k

        # Call method to compute Jacobian function on init
        self.jacobian_reaction_calculate()

    def jacobian_reaction_calculate(self):
        # Define symbols x1 to x9 and k1 to k7
        x_syms = sp.symbols('x1:10')  # x1 to x9
        k_syms = sp.symbols('k1:8')   # k1 to k7

        # Reaction rates
        r38 = k_syms[0] * x_syms[0] * x_syms[1]
        r39 = k_syms[1] * x_syms[0] * x_syms[2]
        r40 = k_syms[2] * x_syms[0] * x_syms[4]
        r41 = k_syms[3] * x_syms[0] * x_syms[5]
        r42 = k_syms[4] * x_syms[0] * x_syms[6]
        r43 = k_syms[5] * x_syms[0] * x_syms[7]
        r44 = k_syms[6] * x_syms[0] * x_syms[8]

        # ODEs
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

        # Jacobian
        J = sp.Matrix(dx).jacobian(x_syms)
        self._x_syms = x_syms  # store symbols for reuse if needed
        self._k_syms = k_syms
        self.J_reaction = sp.lambdify((x_syms, k_syms), J, modules='numpy')

    def jacobian_reaction(self,x_point):
        return self.J_reaction(x_point, self.k)

    def jacobian_observation(self,x):
        H = np.zeros((1, len(self.x)))
        H[0, 3] = 1  # fluoride is the 4rd state (index 3)
        return H
