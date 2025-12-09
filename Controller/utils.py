
# Build MPC once
def estimate_e_with_intensity(constants, c_so3, c_cl, pH, c_pfas_init,k1,intensity):
    """
    Estimate the hydrated electron concentration (eaq-) for given catalyst levels.
    Mirrors the algebra used inside RungeKuttaIntegratorCell.generation_of_eaq().
    Returns a scalar float [M].
    """
    c_oh_m = 10.0 ** (-14.0 + pH)

    i0_185 = intensity * constants["I0_185"]
    i0_254 = intensity * constants["I0_254"]

    # Total absorption @185 nm
    Sigma_f_185 = (constants["epsilon_h2o_185"] * constants["c_h2o"] +
                   constants["epsilon_oh_m_185"] * c_oh_m +
                   constants["epsilon_cl_185"]   * c_cl +
                   constants["epsilon_so3_185"]  * c_so3 +
                   constants["epsilon_pfas_185"] * c_pfas_init)

    # Total absorption @254 nm
    Sigma_f_254 = (constants["epsilon_h2o_254"] * constants["c_h2o"] +
                   constants["epsilon_so3_254"]  * c_so3 +
                   constants["epsilon_pfas_254"] * c_pfas_init)

    # Fractions
    f_h2o_185 = (constants["epsilon_h2o_185"] * constants["c_h2o"]) / Sigma_f_185
    f_oh_m_185 = (constants["epsilon_oh_m_185"] * c_oh_m) / Sigma_f_185
    f_cl_185   = (constants["epsilon_cl_185"]   * c_cl) / Sigma_f_185
    f_so3_185  = (constants["epsilon_so3_185"]  * c_so3) / Sigma_f_185
    f_so3_254  = (constants["epsilon_so3_254"]  * c_so3) / Sigma_f_254

    # Absorbed fractions @185 nm
    term_h2o_185 = f_h2o_185 * constants["phi_h2o_185"] * (1.0 - 10.0 ** (-constants["epsilon_h2o_185"] * constants["l"] * constants["c_h2o"]))
    term_oh_m_185 = f_oh_m_185 * constants["phi_oh_m_185"] * (1.0 - 10.0 ** (-constants["epsilon_oh_m_185"] * constants["l"] * c_oh_m))
    term_cl_185   = f_cl_185   * constants["phi_cl_185"]   * (1.0 - 10.0 ** (-constants["epsilon_cl_185"]   * constants["l"] * c_cl))
    term_so3_185  = f_so3_185  * constants["phi_so3_185"]  * (1.0 - 10.0 ** (-constants["epsilon_so3_185"]  * constants["l"] * c_so3))
    numerator_185 = i0_185* (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

    # Absorbed fraction @254 nm
    numerator_254 = i0_254* f_so3_254 * constants["phi_so3_254"] * (1.0 - 10.0 ** (-constants["epsilon_so3_254"] * constants["l"] * c_so3))

    numerator = float(numerator_185 + numerator_254)
    
    # Denominator (loss terms)
    k_so3_eaq = 1.5e6
    k_cl_eaq  = 1.0e6
    beta_j    = 2.57e4
    denominator = (k1 * c_pfas_init) + beta_j + (k_so3_eaq * c_so3) + (k_cl_eaq * c_cl)

    return numerator / denominator
