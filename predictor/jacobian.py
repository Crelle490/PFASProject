import sympy as sp
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
"""
It may be nessecary to check for observability after calculating the local linearization point
"""
class Jacobian:
    def __init__(self, k, constants, pH, c_cl, c_so3, initial_state=None):
        self.k = np.asarray(k, np.float32)
        self.constants = constants                 # needed by generation_of_eaq()
        self._load_constants(constants)

        self.pH   = float(pH)
        self.c_cl = float(c_cl)
        self.c_so3= float(c_so3)

        # initial state (1×8)
        if initial_state is None:
            self.initial_state = np.array([[1.96e-07, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        else:
            self.initial_state = np.asarray(initial_state, np.float32).reshape(1, 8)

        self.n_states = int(self.initial_state.shape[1])   # 8
        self.y = self.initial_state                        # keep your attribute

        # Hydrated electron generation (scalar)
        numerator = self.generation_of_eaq()

        # Additional kinetic constants
        k_so3_eaq = 1.5e6
        k_cl_eaq  = 1.0e6
        beta_j    = 2.57e4

        # Use initial PFAS concentration (first species)
        c_pfas_init = float(self.initial_state[0, 0])
        k1 = float(self.k[0])
        denominator = k1 * c_pfas_init + beta_j + k_so3_eaq * self.c_so3 + k_cl_eaq * self.c_cl
        self.c_eaq = float(numerator / denominator)

        # Build analytic Jacobian function
        self.jacobian_reaction_calculate()


    # F = df/dy
    def jacobian_reaction_calculate(self):
        # Define symbols y1 to y8 and k1 to k7
        y_syms = sp.symbols('y1:9')  # y1 to y8
        k_syms = sp.symbols('k1:8')   # k1 to k7

        # Reaction rates
        rates = [k * self.c_eaq * y_var for k, y_var in zip(k_syms, y_syms)]

        # Define the ODEs
        dy1 = -rates[0] # C7F15COO⁻
        dy2 = rates[0] - rates[1] # C7F13COO⁻
        dy3 = rates[1] - rates[2] # C7F11COO⁻
        dy4 = rates[2] - rates[3] # C7F9COO⁻
        dy5 = rates[3] - rates[4] # C7F7COO⁻
        dy6 = rates[4] - rates[5] # C7F5COO⁻
        dy7 = rates[5] - rates[6] # C7F#COO⁻
        dy8 = 2 * sum(rates) # F-
        dy = sp.Matrix([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8])

        # ODEs
        #dx[0] = -r38 - r39 - r40 - r41 - r42 - r43 - r44 # C_aq


        # Jacobian
        J = sp.Matrix(dy).jacobian(sp.Matrix(y_syms))
        self._y_syms = y_syms  # store symbols for reuse if needed
        self._k_syms = k_syms
        self.J_reaction = sp.lambdify((y_syms, k_syms), J, modules='numpy')

    def jacobian_reaction(self, y_point):
        A = self.J_reaction(y_point, self.k)
        return np.asarray(A, dtype=np.float32)


    # H = dh/dy FIXXX
    def jacobian_observation(self, y):
        H = np.zeros((1, self.n_states), dtype=np.float32)
        H[0, 7] = 1.0  # fluoride is state 8 (index 7)
        return H



    ####  HELPER FUNCTIONS ####
    def _load_constants(self, constants: dict):
            c = constants
            self.l               = float(c["l"])
            self.I0_185          = float(c["I0_185"])
            self.I0_254          = float(c["I0_254"])
            self.c_h2o           = float(c["c_h2o"])
            self.epsilon_h2o_185 = float(c["epsilon_h2o_185"])
            self.phi_h2o_185     = float(c["phi_h2o_185"])
            self.epsilon_h2o_254 = float(c["epsilon_h2o_254"])
            self.phi_h2o_254     = float(c["phi_h2o_254"])
            self.epsilon_oh_m_185= float(c["epsilon_oh_m_185"])
            self.phi_oh_m_185    = float(c["phi_oh_m_185"])
            self.epsilon_cl_185  = float(c["epsilon_cl_185"])
            self.phi_cl_185      = float(c["phi_cl_185"])
            self.epsilon_so3_185 = float(c["epsilon_so3_185"])
            self.phi_so3_185     = float(c["phi_so3_185"])
            self.epsilon_so3_254 = float(c["epsilon_so3_254"])
            self.phi_so3_254     = float(c["phi_so3_254"])
            self.epsilon_pfas_185= float(c["epsilon_pfas_185"])
            self.epsilon_pfas_254= float(c["epsilon_pfas_254"])

    def generation_of_eaq(self):
        """
        Compute the generation rate of hydrated electrons (e_aq−) from 185/254 nm absorption.
        Returns a scalar (float).
        """
        p = self.constants
        c_pfas_init = float(self.initial_state[0, 0])
        # [OH-] from pH
        c_oh_m = np.power(10.0, -14.0 + self.pH)

        # Total absorption @185 nm
        Sigma_f_185 = (p["epsilon_h2o_185"] * p["c_h2o"] +
                       p["epsilon_oh_m_185"] * c_oh_m +
                       p["epsilon_cl_185"]   * self.c_cl +
                       p["epsilon_so3_185"]  * self.c_so3 +
                       p["epsilon_pfas_185"] * c_pfas_init)

        # Total absorption @254 nm
        Sigma_f_254 = (p["epsilon_h2o_254"] * p["c_h2o"] +
                       p["epsilon_so3_254"]  * self.c_so3 +
                       p["epsilon_pfas_254"] * c_pfas_init)

        # Fractions @185
        f_h2o_185 = (p["epsilon_h2o_185"] * p["c_h2o"]) / Sigma_f_185
        f_oh_m_185 = (p["epsilon_oh_m_185"] * c_oh_m) / Sigma_f_185
        f_cl_185   = (p["epsilon_cl_185"]   * self.c_cl) / Sigma_f_185
        f_so3_185  = (p["epsilon_so3_185"]  * self.c_so3) / Sigma_f_185

        # Fraction @254
        f_so3_254 = (p["epsilon_so3_254"] * self.c_so3) / Sigma_f_254

        # Contributions @185
        term_h2o_185 = f_h2o_185 * p["phi_h2o_185"] * (1.0 - np.power(10.0, -p["epsilon_h2o_185"] * p["l"] * p["c_h2o"]))
        term_oh_m_185 = f_oh_m_185 * p["phi_oh_m_185"] * (1.0 - np.power(10.0, -p["epsilon_oh_m_185"] * p["l"] * c_oh_m))
        term_cl_185   = f_cl_185   * p["phi_cl_185"]   * (1.0 - np.power(10.0, -p["epsilon_cl_185"]   * p["l"] * self.c_cl))
        term_so3_185  = f_so3_185  * p["phi_so3_185"]  * (1.0 - np.power(10.0, -p["epsilon_so3_185"]  * p["l"] * self.c_so3))
        numerator_185 = p["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

        # Contribution @254
        numerator_254 = p["I0_254"] * f_so3_254 * p["phi_so3_254"] * (1.0 - np.power(10.0, -p["epsilon_so3_254"] * p["l"] * self.c_so3))

        return float(numerator_185 + numerator_254)

def build_jacobian_from_config(cfg_dir, k):
    cfg_dir = Path(cfg_dir)
    with open(cfg_dir / "physichal_paramters.yaml", "r") as f:
        constants = yaml.safe_load(f)
    with open(cfg_dir / "initial_conditions.yaml", "r") as f:
        init = yaml.safe_load(f)

    pH   = float(init["pH"])
    c_cl = float(init["c_cl"])
    c_so3 = float(init["c_so3"])
    c_pfas_init = float(init["c_pfas_init"])

    initial_state = np.zeros((1, 8), dtype=np.float32)
    initial_state[0, 0] = c_pfas_init

    return Jacobian(k, constants, pH, c_cl, c_so3, initial_state)

