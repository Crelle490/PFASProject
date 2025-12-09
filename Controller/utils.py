import numpy as np
# Build MPC once
def build_mpc_adi(params: dict,
                  k_list: list | np.ndarray,
                  pH: float,
                  c_pfas_init: float,
                  dt: float,
                  substeps: int,
                  N: int,
                  weights: dict,
                  u_max: list | np.ndarray,
                  x0_flat: np.ndarray,         
                  enable_volume_constraints,
                  du_max: list | np.ndarray | None = None,
                  rk_cell=None,                             # <— NEW
                  pack_state_for_rk=None,                   # <— optional (default: identity)
                  unpack_state_from_rk=None
                  ):               # <— optional (default: identity)

    from helper_functions import make_rhs, build_interval_function, build_single_shoot_nlp, make_normalizers_from_numpy
    """
    Build the CasADi MPC once. Returns a context dict 'ctx' for mpc_adi().
    - params: your YAML 'params' dict (optical/physical constants used by estimate_e)
    - k_list: [k1..k7]
    - pH, c_pfas_init: as in your sim
    - dt, substeps: same integrator resolution the sim uses
    - N: horizon
    - weights: DEFAULT_WEIGHTS-compatible dict
    - u_max: [so3_max, cl_max]
    - x0_flat: initial flat state (8,) for normalizer
    - du_max: per-step rate limits; default = 0.2 * u_max
    - monotonic: enforce uk >= uk_prev if True
    """
    print("Building CasADi MPC context...") 

    k_arr = np.asarray(k_list, dtype=float)
    u_max_arr = np.asarray(u_max, dtype=float)
    if du_max is None:
        du_max_arr = 0.2 * u_max_arr
    else:
        du_max_arr = np.asarray(du_max, dtype=float)

    # Symbolic RHS and one-interval map Phi (same dt/substeps as plant)
    rhs = make_rhs(params, k_arr, pH, c_pfas_init)
    Phi = build_interval_function(dt=dt, substeps=int(substeps), f_rhs=rhs)

    # Normalizers for cost scaling
    z_scale, u_scale = make_normalizers_from_numpy(x0_flat, u_max_arr)

    # Build the NLP solver
    solver, pack_p, unpack_u, lbx, ubx, lbg, ubg = build_single_shoot_nlp(
        Phi=Phi,
        N=int(N),
        weights=weights,
        #z_scale=z_scale,
        #u_scale=u_scale,
        u_max=u_max_arr,
        du_max=du_max_arr,
        c_pfas_init=c_pfas_init,
        enable_volume_constraints=enable_volume_constraints,
    )

        # ---- Attach numeric integrator info for live_plotter (no CasADi needed there) ----
    if pack_state_for_rk is None:
        pack_state_for_rk = lambda x: np.asarray(x, float).reshape(-1)
    if unpack_state_from_rk is None:
        unpack_state_from_rk = lambda s: np.asarray(s, float).reshape(-1)



    ctx = {
        "solver": solver,
        "pack_p": pack_p,
        "unpack_u": unpack_u,
        "lbx": lbx,
        "ubx": ubx,
        "lbg": lbg,
        "ubg": ubg,
        "N": int(N),
        "z_scale": z_scale,
        "u_scale": u_scale,
        "Phi": Phi,
    }
    ctx.update({
        "rk_cell": rk_cell,                                  # used by advance_one_control_step
        "substeps": int(substeps),                           # dt * substeps = Ts
        "pack_state_for_rk": pack_state_for_rk,              # optional adapters
        "unpack_state_from_rk": unpack_state_from_rk,
    })

    # Attach numeric integrator (for live_plotter horizon via advance_one_control_step)
    if pack_state_for_rk is None:
        pack_state_for_rk = lambda x: np.asarray(x, float).reshape(-1)
    if unpack_state_from_rk is None:
        unpack_state_from_rk = lambda s: np.asarray(s, float).reshape(-1)

    ctx.update({
        "rk_cell": rk_cell,
        "substeps": int(substeps),
        "pack_state_for_rk": pack_state_for_rk,
        "unpack_state_from_rk": unpack_state_from_rk,
    })

    print("CasADi MPC context built.")
    return ctx


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
