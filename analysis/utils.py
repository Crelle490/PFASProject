import tensorflow as tf
import numpy as np

# Kinetic constants
K_SO3_EAQ = 1.5e6
K_CL_EAQ = 1.0e6
BETA_J = 2.57e4
FLUORIDE_STOICHIOMETRY = 2.0
NUM_PFAS_SPECIES = 7


def _fun(self, y, params):
    """
    Compute PFAS degradation rates.
    
    Args:
        y: Batch of state variables (batch, 8) - first 7 are PFAS species, last is fluoride
        params: Dictionary of kinetic parameters
        
    Returns:
        Concatenated rate derivatives for all species
    """
    # Extract PFAS species (first 7 components)
    y_vars = [y[:, i:i+1] for i in range(NUM_PFAS_SPECIES)]

    # Hydrated electron generation rate
    c_eaq_numerator = self.generation_of_eaq()

    # Calculate hydrated electron concentration
    c_pfas_init = float(self.initial_state[0, 0])
    c_eaq_denominator = (
        params['k1'] * c_pfas_init
        + BETA_J
        + K_SO3_EAQ * self.c_so3
        + K_CL_EAQ * self.c_cl
    )
    c_eaq = c_eaq_numerator / c_eaq_denominator

    # Compute reaction rates (first-order in PFAS and c_eaq)
    rates = [params[f'k{i+1}'] * c_eaq * y_vars[i] for i in range(NUM_PFAS_SPECIES)]

    # Construct rate derivatives (cascade: each species converts to next)
    dy = [-rates[0]]
    for i in range(NUM_PFAS_SPECIES - 1):
        dy.append(rates[i] - rates[i + 1])
    
    # Fluoride production (stoichiometry = 2.0 per reaction)
    dy.append(FLUORIDE_STOICHIOMETRY * tf.reduce_sum(rates, axis=0))

    return tf.concat(dy, axis=-1)


def generation_of_eaq(self):
    """
    Compute the generation rate of hydrated electrons (e_aq−) from 185/254 nm UV absorption.
    
    Uses beer-lambert law with quantum yields for different absorbing species.
    
    Returns:
        float: Generation rate of hydrated electrons (mol/s or similar units)
    """
    p = self.constants
    c_pfas_init = float(self.initial_state[0, 0])
    c_oh_m = np.power(10.0, -14.0 + self.pH)

    # ========== 185 nm Absorption ==========
    sigma_185 = (
        p["epsilon_h2o_185"] * p["c_h2o"]
        + p["epsilon_oh_m_185"] * c_oh_m
        + p["epsilon_cl_185"] * self.c_cl
        + p["epsilon_so3_185"] * self.c_so3
        + p["epsilon_pfas_185"] * c_pfas_init
    )

    # Calculate molar fractions at 185 nm
    f_h2o_185 = (p["epsilon_h2o_185"] * p["c_h2o"]) / sigma_185
    f_oh_m_185 = (p["epsilon_oh_m_185"] * c_oh_m) / sigma_185
    f_cl_185 = (p["epsilon_cl_185"] * self.c_cl) / sigma_185
    f_so3_185 = (p["epsilon_so3_185"] * self.c_so3) / sigma_185

    # ========== 254 nm Absorption ==========
    sigma_254 = (
        p["epsilon_h2o_254"] * p["c_h2o"]
        + p["epsilon_so3_254"] * self.c_so3
        + p["epsilon_pfas_254"] * c_pfas_init
    )

    f_so3_254 = (p["epsilon_so3_254"] * self.c_so3) / sigma_254

    # ========== Beer-Lambert Contributions @185 nm ==========
    term_h2o_185 = _compute_absorption_term(
        f_h2o_185, p["phi_h2o_185"], p["epsilon_h2o_185"], p["c_h2o"], p["l"]
    )
    term_oh_m_185 = _compute_absorption_term(
        f_oh_m_185, p["phi_oh_m_185"], p["epsilon_oh_m_185"], c_oh_m, p["l"]
    )
    term_cl_185 = _compute_absorption_term(
        f_cl_185, p["phi_cl_185"], p["epsilon_cl_185"], self.c_cl, p["l"]
    )
    term_so3_185 = _compute_absorption_term(
        f_so3_185, p["phi_so3_185"], p["epsilon_so3_185"], self.c_so3, p["l"]
    )

    numerator_185 = p["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

    # ========== Beer-Lambert Contribution @254 nm ==========
    numerator_254 = (
        p["I0_254"]
        * f_so3_254
        * p["phi_so3_254"]
        * (1.0 - np.power(10.0, -p["epsilon_so3_254"] * p["l"] * self.c_so3))
    )

    return float(numerator_185 + numerator_254)


def _compute_absorption_term(fraction, quantum_yield, extinction, concentration, path_length):
    """
    Compute absorption term for beer-lambert law: f * phi * (1 - 10^(-e*l*c))
    
    Args:
        fraction: Molar fraction of absorbing species
        quantum_yield: Quantum yield (phi)
        extinction: Extinction coefficient (epsilon)
        concentration: Molar concentration
        path_length: Path length (l)
        
    Returns:
        float: Absorption term contribution
    """
    optical_depth = extinction * path_length * concentration
    return fraction * quantum_yield * (1.0 - np.power(10.0, -optical_depth))

