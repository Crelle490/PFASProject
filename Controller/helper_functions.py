from pathlib import Path
import yaml
import numpy as np
from copy import deepcopy
# CasADi MPC builders
from casadi_mpc import (
    make_rhs,
    build_interval_function,
    make_normalizers_from_numpy,
    build_single_shoot_nlp,
    NU
)

DEFAULT_WEIGHTS = {
    "qx": 100.0,
    "qf": 3*100.0,               # qf = N * qx terminal focus same strength as full horizon. not too large, or it will still push to max
    "R":  np.array([69.3, 14.6, 0.0,0.01]),   # weights for [so3, cl, pH, intensity] given in DKK/mol, DKK/mol, DKK/unit pH, DKK/intensity
    "Rd": np.array([0.0, 0.0,0.0045,0.0]),  # maybe just delete. smooth changes but not too restrictive
    "eps": 1e-10,
    "taus": np.array([0.50, 0.3, 0.25, 0.2, 0.1, 0.05, 0.02]), # thresholds for PFAS species
    "sharp": 0.4,  # sharpness for priority weights
    "q_sum": 0.05,  # weight for sum of PFAS in lex cost
    "qf_sum": 0.2*3*0.05, # qf_sum = 0.2 * N * q_sum. weight for sum of PFAS in
}





# Find root directory
def find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "config").is_dir():
            return p
    return start

# Load parameters from YAML files
def load_yaml_params(cfg_dir: Path):
    phys_candidates = [
        cfg_dir / "physichal_paramters.yaml",
        cfg_dir / "physical_parameters.yaml",
        cfg_dir / "physical_paramters.yaml",
    ]
    phys_path = next((p for p in phys_candidates if p.exists()), None)
    if phys_path is None:
        raise FileNotFoundError(f"Could not find any of: {', '.join(str(p) for p in phys_candidates)}")
    with open(phys_path, "r") as f:
        params = yaml.safe_load(f)

    init_path = cfg_dir / "initial_conditions.yaml"
    if not init_path.exists():
        raise FileNotFoundError(f"Missing required file: {init_path}")
    with open(init_path, "r") as f:
        init_vals = yaml.safe_load(f)
    return params, init_vals

# Load trained kinetic constants
def load_yaml_constants(cfg_dir: Path):
    phys_path = cfg_dir / 'trained_params.yaml'
    if phys_path is None:
        raise FileNotFoundError(f"Could not find trained parameters file: {phys_path}")
    with open(phys_path, "r") as f:
        params = yaml.safe_load(f)

    return params

# Load covariance
def load_yaml_covariance(cfg_dir=None):
    """
    Load covariance parameters YAML.

    Accepts:
      - None → uses <project_root>/config/covariance_params.yaml
      - str or Path → either a directory containing the YAML file,
                      or a direct path to the YAML file itself
    Returns
    -------
    dict : parsed YAML contents
    """
    # --- ensure Path type ---
    if cfg_dir is None:
        # fallback: current working dir + config subdir
        cfg_dir = Path.cwd() / "config"
    else:
        cfg_dir = Path(cfg_dir)

    # --- determine file path ---
    if cfg_dir.is_file():
        phys_path = cfg_dir
    else:
        phys_path = cfg_dir / "covariance_params.yaml"

    # --- verify existence ---
    if not phys_path.exists():
        raise FileNotFoundError(f"Could not find covariance YAML at: {phys_path}")

    # --- load YAML ---
    with open(phys_path, "r") as f:
        params = yaml.safe_load(f)

    return params


# Unused PID Controller

def pid(state,dt=1.0,concentraction_of_so3=10e-6,concentraction_of_cl=10e-6):
    target = 10**(-10)
    err = state - target
    Kp = np.array([1.0,1.5])  # proportional gain [so3, cl]
    control = Kp * err

    # Clamp control signal
    control = np.maximum(control, 0.0)

    # Determine if control signal is physically possible
    time_in_sec = dt/60.0
    max_flow_rate = 11.0/(time_in_sec)  # maximum flow rate of catalyst [ml/s]
    max_so3_flow = max_flow_rate*concentraction_of_so3
    max_cl_flow = max_flow_rate*concentraction_of_cl
    max_control_signal = [max_so3_flow, max_cl_flow]

    # Flow constraint enforcement
    for i in range(len(control)):
        if control[i] > max_control_signal[i]:
            control[i] = max_control_signal[i]

    return control.astype(np.float32)


# Compute hydrated electrons
def estimate_e(constants, c_so3, c_cl, pH, c_pfas_init, k1):
    """
    Estimate the hydrated electron concentration (eaq-) for given catalyst levels.
    Mirrors the algebra used inside RungeKuttaIntegratorCell.generation_of_eaq().
    Returns a scalar float [M].
    """
    c_oh_m = 10.0 ** (pH-14.0)

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
    numerator_185 = constants["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

    # Absorbed fraction @254 nm
    numerator_254 = constants["I0_254"] * f_so3_254 * constants["phi_so3_254"] * (1.0 - 10.0 ** (-constants["epsilon_so3_254"] * constants["l"] * c_so3))

    numerator = float(numerator_185 + numerator_254)

    # Denominator (loss terms)
    k_so3_eaq = 1.5e6
    k_cl_eaq  = 1.0e6
    beta_j    = 2.57e4
    denominator = (k1 * c_pfas_init) + beta_j + (k_so3_eaq * c_so3) + (k_cl_eaq * c_cl)

    return numerator / denominator

# Choose Ts - not used yet
def choose_Ts_fixed(k_list, e_max, L, T_sens,
                    safety_rxn=10.0, safety_tr=5.0, dt_int=10.0):
    """
    Compute a fixed controller sampling time Ts that is safe for all expected conditions.

    Parameters
    ----------
    k_list : list or array
        Reaction rate constants [s^-1 per M].
    e_max : float
        Maximum expected hydrated electron concentration [M].
    L : float
        Loop residence/transport time [s].
    T_sens : float
        Sensor update period [s].
    safety_rxn : float
        Safety factor on reaction speed (default 10).
    safety_tr : float
        Safety factor on transport time (default 5).
    dt_int : float
        Internal integration step [s].

    Returns
    -------
    Ts : float
        Fixed sampling time for MPC [s].
    substeps : int
        Integer number of RK steps per control interval.
    """
    k_max = float(max(k_list))
    Ts_rxn = 1.0 / (safety_rxn * k_max * e_max)  # respect fastest kinetics
    Ts_tr  = float(L) / safety_tr                # respect transport
    Ts     = min(Ts_rxn, Ts_tr, float(T_sens))
    substeps = max(1, int(round(Ts / float(dt_int))))
    Ts = substeps * float(dt_int)  # snap to integer multiple of dt_int
    return Ts, substeps


def advance_one_control_step(rk_cell, xk, uk, substeps):
    """
    Advance the plant by Ts = substeps * dt_int with input uk held constant.
    xk : state tensor/array shaped like rk_cell expects (e.g., (1,1,8) or (1,8))
    uk : [so3, cl]
    returns x_{k+1} with same shape convention as input
    """
    states = xk
    so3, cl, pH, intensity = float(uk[0]), float(uk[1]), float(uk[2]), float(uk[3])
    #print(f"advance_one_control_step so3: {so3}, cl: {cl}")
    for _ in range(int(substeps)):
        _, states = rk_cell.call(inputs=[so3, cl,pH,intensity], states=states)
    return states

def vol_from_deltaC_safe(deltaC, Cc, Vs, eps=1e-12):
    """
    Vc = (ΔC_eff * Vs) / (Cc - ΔC_eff),  with
    ΔC_eff = smooth positive part of ΔC, softly capped to < alpha*Cc.
    This is differentiable at ΔC=0 and stays away from the pole ΔC→Cc.
    """
    dv = (deltaC * Vs) / (Cc - deltaC + eps) 
    return np.max([dv, 0.0])

 # ---- MPC WITH CasADI ----

# Build MPC once
def build_mpc_adi(params: dict,
                  k_list: list | np.ndarray,
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
    rhs = make_rhs(params, k_arr, c_pfas_init) # Okay
    Phi = build_interval_function(dt=dt, substeps=int(substeps), f_rhs=rhs) # Okay

    # Normalizers for cost scaling
    z_scale, u_scale = make_normalizers_from_numpy(x0_flat, u_max_arr) # Okay
 
    # Build the NLP solver
    solver, pack_p, unpack_u, lbx, ubx, lbg, ubg = build_single_shoot_nlp(
        Phi=Phi,
        N=int(N),
        weights=weights,
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


# func to run one MPC step in loop
def mpc_adi(xk_flat: np.ndarray,
            uk_prev: np.ndarray,
            ctx: dict,
            Vs0,
            V_sens,
            V_max,
            C_c,
            #taus: np.ndarray,
            warm_start: np.ndarray | None = None):
    """
    Solve one MPC step with the CasADi solver built in build_mpc_adi().
    Inputs:
      - xk_flat: current state (8,) flat
      - uk_prev: previous input (2,)
      - ctx: dict from build_mpc_adi()
      - warm_start: optional initial guess for the stacked controls (length = N*2).
                    If None, repeats uk_prev across the horizon.

    Returns:
      - u_first: (2,) control to apply now
      - U_star:  list of N arrays (each length 2)
      - J_star:  float optimal cost
    """
    solver   = ctx["solver"]
    pack_p   = ctx["pack_p"]
    unpack_u = ctx["unpack_u"]
    lbx, ubx = ctx["lbx"], ctx["ubx"]
    lbg, ubg = ctx["lbg"], ctx["ubg"]
    z_scale  = ctx["z_scale"]
    u_scale  = ctx["u_scale"]
    N        = ctx["N"]
    # parameter vector
    p_vec = pack_p(xk_flat, uk_prev, z_scale, u_scale,Vs0,V_sens,V_max,C_c)

    # initial guess
    if warm_start is None or np.size(warm_start) != N*2:
        U_init = np.tile(np.asarray(uk_prev, float), N)
    else:
        U_init = np.asarray(warm_start, float).ravel()

    # solve
    sol = solver(x0=U_init, p=p_vec, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    #sol = solver(x0=U_init, p=p_vec, lbg=lbg, ubg=ubg)
    U_star_vec = np.array(sol["x"]).ravel()
    J_star = float(sol["f"])
    U_star = unpack_u(U_star_vec)
    u_first = np.asarray(U_star[0], dtype=float)

    # Clamp all outputs to solver bounds to avoid any numerical spillover
    lb = np.asarray(ctx["lbx"][:NU], float)  # first step bounds
    ub = np.asarray(ctx["ubx"][:NU], float)
    u_first = np.clip(u_first, lb, ub)
    U_star = [np.clip(np.asarray(u, float), lb, ub) for u in U_star]


    print(f"  MPC CasADi solve complete - J*: {J_star:.4f}")

    return u_first, U_star, J_star
