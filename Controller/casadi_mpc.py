# casadi_mpc.py
import casadi as ca
import numpy as np
from typing import Callable

# ---- 0) Dimensions (match your system) ----
NX = 8   # states
NU = 2   # inputs: [so3, cl]

# 1) --- Build symbolic dynamics --- 
# # symbolic f(x,u) expression
def make_rhs(P: dict, k_list: np.ndarray, pH: float, c_pfas_init: float):
    """
    P: dict of optical/physical constants (same keys as your YAML / estimate_e)
    k_list: array-like of [k1..k7]
    pH: scalar
    c_pfas_init: scalar (your 'c_pfas_init' used in estimate_e denominator)
    """
    # small eps to avoid division by zero in fractions
    eps = 1e-30

    def estimate_e_sym(c_so3, c_cl):
        """Symbolic mirror of estimate_e(...), CasADi-safe ops."""
        c_oh_m = ca.power(10.0, -14.0 + pH)

        # --- totals @185 nm ---
        Sigma_f_185 = (P["epsilon_h2o_185"] * P["c_h2o"]
                     + P["epsilon_oh_m_185"] * c_oh_m
                     + P["epsilon_cl_185"]   * c_cl
                     + P["epsilon_so3_185"]  * c_so3
                     + P["epsilon_pfas_185"] * c_pfas_init) + eps

        # --- totals @254 nm ---
        Sigma_f_254 = (P["epsilon_h2o_254"] * P["c_h2o"]
                     + P["epsilon_so3_254"]  * c_so3
                     + P["epsilon_pfas_254"] * c_pfas_init) + eps

        # fractions
        f_h2o_185 = (P["epsilon_h2o_185"] * P["c_h2o"]) / Sigma_f_185
        f_oh_m_185 = (P["epsilon_oh_m_185"] * c_oh_m)   / Sigma_f_185
        f_cl_185   = (P["epsilon_cl_185"]   * c_cl)     / Sigma_f_185
        f_so3_185  = (P["epsilon_so3_185"]  * c_so3)    / Sigma_f_185
        f_so3_254  = (P["epsilon_so3_254"]  * c_so3)    / Sigma_f_254

        # absorbed @185
        term_h2o_185 = f_h2o_185 * P["phi_h2o_185"] * (1 - ca.power(10.0, -P["epsilon_h2o_185"] * P["l"] * P["c_h2o"]))
        term_oh_m_185 = f_oh_m_185 * P["phi_oh_m_185"] * (1 - ca.power(10.0, -P["epsilon_oh_m_185"] * P["l"] * c_oh_m))
        term_cl_185   = f_cl_185   * P["phi_cl_185"]   * (1 - ca.power(10.0, -P["epsilon_cl_185"]   * P["l"] * c_cl))
        term_so3_185  = f_so3_185  * P["phi_so3_185"]  * (1 - ca.power(10.0, -P["epsilon_so3_185"]  * P["l"] * c_so3))
        numerator_185 = P["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

        # absorbed @254 (only SO3 path)
        numerator_254 = P["I0_254"] * f_so3_254 * P["phi_so3_254"] * (1 - ca.power(10.0, -P["epsilon_so3_254"] * P["l"] * c_so3))

        numerator = numerator_185 + numerator_254

        # denominator (loss terms) — constants as in your Python
        k_so3_eaq = 1.5e6
        k_cl_eaq  = 1.0e6
        beta_j    = 2.57e4
        denominator = (k_list[0] * c_pfas_init) + beta_j + (k_so3_eaq * c_so3) + (k_cl_eaq * c_cl) + eps

        return numerator / denominator

    k_vec = [float(ki) for ki in k_list]


    def rhs_sym(x: ca.SX, u: ca.SX) -> ca.SX:
        """
        Symbolic right-hand side of the PFAS degradation ODE system.

        Parameters
        ----------
        x : ca.SX
            State vector [PFAS1, PFAS2, PFAS3, PFAS4, PFAS5, PFAS6, PFAS7, F_minus].
        u : ca.SX
            Input vector [SO3-, Cl-].

        Returns
        -------
        ca.SX
            Time derivative dx/dt as CasADi SX(8x1)
        """

        # ---------- unpack inputs ----------
        c_so3 = u[0]     # SO3 concentration
        c_cl  = u[1]     # Cl- concentration


        # individual PFAS species (7)
        c_pfas1 = x[0]
        c_pfas2 = x[1]
        c_pfas3 = x[2]
        c_pfas4 = x[3]
        c_pfas5 = x[4]
        c_pfas6 = x[5]
        c_pfas7 = x[6]
        c_f     = x[7]   # fluoride (F-)

        # ---------- hydrated electron concentration ----------
        e = estimate_e_sym(c_so3, c_cl)  # symbolic scalar

        # ---------- explicit differential equations ----------
        # dx1/dt = -k1 * e * PFAS1
        dx1 = -k_vec[0] * e * c_pfas1

        # dx2/dt = +k1 * e * PFAS1 - k2 * e * PFAS2
        dx2 =  k_vec[0] * e * c_pfas1 - k_vec[1] * e * c_pfas2

        # dx3/dt = +k2 * e * PFAS2 - k3 * e * PFAS3
        dx3 =  k_vec[1] * e * c_pfas2 - k_vec[2] * e * c_pfas3

        # dx4/dt = +k3 * e * PFAS3 - k4 * e * PFAS4
        dx4 =  k_vec[2] * e * c_pfas3 - k_vec[3] * e * c_pfas4

        # dx5/dt = +k4 * e * PFAS4 - k5 * e * PFAS5
        dx5 =  k_vec[3] * e * c_pfas4 - k_vec[4] * e * c_pfas5

        # dx6/dt = +k5 * e * PFAS5 - k6 * e * PFAS6
        dx6 =  k_vec[4] * e * c_pfas5 - k_vec[5] * e * c_pfas6

        # dx7/dt = +k6 * e * PFAS6 - k7 * e * PFAS7
        dx7 =  k_vec[5] * e * c_pfas6 - k_vec[6] * e * c_pfas7

        # Fluoride formation rate = 2 * sum(k_i * e * PFAS_i)
        # (the factor 2 assumes two fluorine atoms released per PFAS molecule)
        dxF = 2 * (
            k_vec[0] * e * c_pfas1 +
            k_vec[1] * e * c_pfas2 +
            k_vec[2] * e * c_pfas3 +
            k_vec[3] * e * c_pfas4 +
            k_vec[4] * e * c_pfas5 +
            k_vec[5] * e * c_pfas6 +
            k_vec[6] * e * c_pfas7
        )

        # ---------- stack into a single SX vector ----------
        xdot = ca.vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7, dxF)
        return xdot


    return rhs_sym

# Symbolic RK4 substep
def rk4_substep(f: Callable[[ca.SX, ca.SX], ca.SX], x: ca.SX, u: ca.SX, dt: float) -> ca.SX:
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3,     u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Advance over multiple substeps: Ts = substeps * dt
def rk4_interval_map(f: Callable[[ca.SX, ca.SX], ca.SX], x0: ca.SX, u: ca.SX, dt: float, substeps: int) -> ca.SX:
    x = x0
    for _ in range(int(substeps)):
        x = rk4_substep(f, x, u, dt)
    return x

def dilution_factor(du: ca.DM, Vs: ca.DM) -> ca.DM:
    """ Compute dilution factor over one control interval. """
    gamma = 1 - du / (Vs + du)
    return gamma

# Build map x_next = Phi(x,u) over one control interval Ts
def build_interval_function(dt: float, substeps: int, f_rhs) -> ca.Function:
    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', NU)
    x_next = rk4_interval_map(f_rhs, x, u, dt, substeps)
    return ca.Function('Phi', [x, u], [x_next], {'jit': True})

# 2) --- Cost helpers (CasADi-friendly) ---

# sum of PFAS species
def z_pfas_sym(x: ca.SX) -> ca.SX:
    """Sum of PFAS species x[0:7] (scalar SX)."""
    return ca.sum1(x[0:7])

# normalize states and inputs for cost calculation
def make_normalizers_from_numpy(x0_flat: np.ndarray, u_max: np.ndarray):
    ratio = np.array([1, 0.6558, 0.1544, 0.0582, 0.0976, 0.0949, 0.1479, 14.0]) # ratio order of magnitude
    init  = float(x0_flat[0]) # initial PFAS_1 concentration

    z_scales = init * ratio # scale vector for states
    z_floor  = 1e-9   
    z_scales = np.maximum(z_scales, z_floor)

    u_scale = np.asarray(u_max, dtype=float)


    print(f"Normalizers - z_scale: {z_scales}, u_scale: {u_scale}")
    return z_scales, u_scale


# dynamic cost to proiterize current state reduction
def _priority_weights_sym(x, taus,z_scale, sharp, eps=1e-12):
    """
    Symbolic, smooth 'first-not-done' weights over PFAS_1..PFAS_7.
    x    : (NX x 1) MX/SX; we read x[0:7] as PFAS species
    taus : (7,) or (7x1) vector of thresholds (DM or MX/SX)
    sharp: sigmoid sharpness (bigger -> crisper switching)
    """
    x7    = x[0:7]/z_scale[0:7]                    # (7x1)
    taus  = ca.reshape(taus, 7, 1)       # (7x1)
    denom = sharp*taus + eps             # (7x1)
    #denom = np.ones_like(denom)

    # gates ~1 when xi > tau_i, ~0 when xi <= tau_i
    z     = (x7 - taus)/denom
    g     = 1.0 / (1.0 + ca.exp(-z))     # (7x1)

    # focus_i = prod_{k<i}(1-g_k) * g_i
    focus = ca.MX.zeros(7, 1)
    prod_done = 1.0
    for i in range(7):
        focus[i] = prod_done * g[i]
        prod_done = prod_done * (1.0 - g[i])

    # normalize to sum ~1 (keeps gradients well-scaled)
    w = focus + eps
    
    return w / (ca.sum1(w) + eps)        # (7x1)

# func to compute stage cost
def stage_cost_sym_lex(xk, uk, uk_prev,
                       qx: float, R: np.ndarray, Rd: np.ndarray,
                       z_scale, u_scale,
                       taus, sharp: float , q_sum: float, c_pfas_init: float):
    """
    Lexicographic stage cost:
      - focus on the FIRST not-finished PFAS species (1..7)
      - optional small pressure on total ΣPFAS via q_sum
    Keeps your original signature (qx,R,Rd,...) and adds:
      - taus : 7 thresholds (same units as x)
      - sharp: sigmoid sharpness (20–80 is a good range)
      - q_sum: small weight on total PFAS (e.g. 0.05)
      - c_pfas_init: initial total PFAS (for final target calculation)
    """
    # Cast constants
    R_c   = ca.DM(R).reshape((NU, 1))        # (2x1)
    Rd_c  = ca.DM(Rd).reshape((NU, 1))       # (2x1)
    #us_c  = ca.DM(u_scale).reshape((NU, 1))  # (2x1)
    us_c = u_scale
    zsc_c = z_scale + 1e-30                  # scalar MX/SX guard

    # Inputs (normalized)
    u_norm  = uk / us_c
    du_norm = (uk - uk_prev) / us_c
    u_quad  = ca.sum1(R_c  * (u_norm**2))
    du_quad = ca.sum1(Rd_c * (du_norm**2))

    # Dynamic lexicographic weights over PFAS_1..PFAS_7
    #w_dyn = _priority_weights_sym(xk, taus,z_scale, sharp=sharp)   # (7x1)
    w_dyn = ca.DM.ones(7,1) # don't use priority weights for now

    # Focused PFAS penalty: sum_i w_i * (x_i/z)^2 over i=1..7
    x7 = xk[0:7] / zsc_c[0:7] 

    # Penalize F- when away from final target (not used right now)
    L_focus = qx * ca.sum1(w_dyn * (x7**2))
    F_final = c_pfas_init*2*(NX-1)
    L_sum = 0 #q_sum * ((F_final-xk[-1]) / zsc_c[-1])**2

    return L_focus + L_sum + u_quad + du_quad



# func to compute terminal cost
def terminal_cost_sym_lex(xN, qf: float, z_scale, taus, sharp, qf_sum,c_pfas_init):
    """
    Terminal cost with the same priority logic.
      - qf: weight on the focused species at terminal state
      - qf_sum: optional small terminal pressure on total ΣPFAS
    """
    zsc_c = z_scale + 1e-30
    #w_dyn_N = _priority_weights_sym(xN, taus,z_scale, sharp=sharp,)
    w_dyn_N = np.ones((7,)) # don't use priority weights for now

    # Termial cost of PFAS specis
    x7N = xN[0:7] / zsc_c[0:7]
    L_focus_N = qf * ca.sum1(w_dyn_N * (x7N**2))


    # Penalize F- when terminal away from final target (not used right now)
    F_final = c_pfas_init*2*(NX-1)
    L_sum_N   = 0 #qf_sum * ((xN[-1]-F_final)/ zsc_c[-1])**2

    return L_focus_N + L_sum_N


def _vol_from_deltaC(uk: ca.MX,up: ca.MX, Cx: ca.MX, Vs: ca.MX, eps: float = 1e-12) -> ca.MX:

    dV = Vs*(up - uk)/(uk-Cx)
    return dV

def smooth_pos(x, eps=1e-12):
    # C¹ approx to max(x,0)
    return 0.5*(x + ca.sqrt(x*x + eps))


def vol_from_deltaC_safe(deltaC, Cc, Vs, eps=1e-12):
    """
    Vc = (ΔC_eff * Vs) / (Cc - ΔC_eff),  with
    ΔC_eff = smooth positive part of ΔC, softly capped to < alpha*Cc.
    This is differentiable at ΔC=0 and stays away from the pole ΔC→Cc.
    """
    dv = (deltaC * Vs) / (Cc - deltaC + eps)
    return ca.fmax(dv, 0.0)

# 3) --- NLP SOLVER -----
def build_single_shoot_nlp(Phi: ca.Function,
                           N: int,
                           weights: dict,
                           #z_scale: np.ndarray,
                           #u_scale: np.ndarray,
                           u_max: np.ndarray,
                           du_max: np.ndarray,
                           c_pfas_init: float,
                           enable_volume_constraints: bool
                           ):
    #  decision variables: stacked U = [u0, u1, ..., u_{N-1}]. 
    V = ca.MX.sym('U', N*NU) # varying parameters of 2*N
    U_k = lambda k: V[k*NU:(k+1)*NU]  # view kth control

    #  parameters: not varying but symbolic inputs to solver
    x0      = ca.MX.sym('x0', NX) # initial state
    u_prev0 = ca.MX.sym('u_prev0', NU) # previous control
    zsc     = ca.MX.sym('zsc', NX)     # state normalizer
    uscale  = ca.MX.sym('uscale', NU)   # vector used to normalize inputs
    Vs0      = ca.MX.sym('Vs0')           # current solution volume
    V_sens  = ca.MX.sym('V_sens')         # sampling volume
    V_max   = ca.MX.sym('V_max')          # maximum volume
    C_c     = ca.MX.sym('C_c',NU)            # catalyst feed concentrations
    #taus    = ca.MX.sym("taus",7)    #u_max     = ca.MX.sym('u_max',NU)            # catalyst feed concentrations



    qx = float(weights["qx"]) 
    qf = float(weights["qf"])
    R  = np.asarray(weights["R"],  dtype=float)
    Rd = np.asarray(weights["Rd"], dtype=float)
    #taus = np.asarray(weights["taus"], dtype=float)
    sharp = float(weights["sharp"])
    q_sum = float(weights["q_sum"])
    qf_sum = float(weights["qf_sum"])
    
    taus = ca.DM.ones(7,1)


    # Function of J(u; x): symbolic function to minimize
    J  = ca.MX(0)
    xk = x0
    up = u_prev0
    Vs = Vs0
    Cc1 = C_c[0]
    Cc2 = C_c[1]
    for k in range(N):
        uk = U_k(k)
        dC = uk - up
        dC1 = dC[0]
        dC2 = dC[1]
        Vc1 = vol_from_deltaC_safe(dC1, Cc1, Vs, eps=1e-12)
        Vc2 = vol_from_deltaC_safe(dC2, Cc2, Vs, eps=1e-12)

        #  Total volume after addition
        Vs = Vs - V_sens # update volume after sensing

        Vsum = Vc1 + Vc2 # total added volume due to catalyst addition
        gamma = dilution_factor(Vsum, Vs)
        
        Vs = Vs + Vsum # update volume after addition

        xk = xk*gamma # apply dilution to state
        J += stage_cost_sym_lex(xk, uk, up, qx, R, Rd, zsc, uscale, taus, sharp, q_sum,c_pfas_init=c_pfas_init) # stage cost
        xk = Phi(xk, uk)  # x_{k+1}
        up = uk
    
    J += terminal_cost_sym_lex(xk, qf, zsc, taus, sharp, qf_sum,c_pfas_init=c_pfas_init) # terminal cost
    
     
    # Create constraints vector G and bounds lbg, ubg
        # Create constraints vector G and bounds lbg, ubg
    g = []  # constraint expressions
    lbg = []
    ubg = []

    # Use PHYSICAL units here (no normalization)
    up  = u_prev0        # previous input (physical [M])
    eps = 1e-12

    Vs = Vs0             # current working volume [same units as Vs0, V_sens, V_max]
    Cc1 = C_c[0]         # SO3 stock concentration [M]
    Cc2 = C_c[1]         # Cl stock concentration  [M]

    for k in range(N):
        uk = U_k(k)      # (2x1), PHYSICAL input [M]

        # change in working-solution concentration this step
        dC  = uk - up
        dC1 = dC[0]
        dC2 = dC[1]

        # volume AFTER sampling (same structure as in simulate())
        Vs_before = Vs - V_sens

        # volume added by each catalyst (physical units)
        Vc1 = vol_from_deltaC_safe(dC1, Cc1, Vs_before, eps=eps)
        Vc2 = vol_from_deltaC_safe(dC2, Cc2, Vs_before, eps=eps)
        Vsum = Vc1 + Vc2

        # maximum free volume available this step
        dV_max_k = V_max - Vs_before  # [volume units]

        # 1) total added volume constraint:  0 <= Vsum <= dV_max_k
        #    -> normalized as 0 <= Vsum / dV_max_k <= 1
        g.append(Vsum / dV_max_k)
        lbg += [0.0]
        ubg += [1.0]

        # 2) per-catalyst "feasible ΔC" constraints (optional)
        # maximum moles you could still add without exceeding V_max
        dn_max1 = dV_max_k * Cc1   # [moles]
        dn_max2 = dV_max_k * Cc2

        # update volume AFTER dosing
        Vs_after = Vs_before + Vsum

        # corresponding maximum ΔC in working solution
        dU_max1 = dn_max1 / Vs_after
        dU_max2 = dn_max2 / Vs_after

        # avoid division by zero (if Vs_after very tiny)
        # you can add a safety clamp if needed:
        # dU_max1 = ca.fmax(dU_max1, eps)
        # dU_max2 = ca.fmax(dU_max2, eps)

        g.append(dC1 / dU_max1)
        lbg += [0.0]
        ubg += [1.0]

        g.append(dC2 / dU_max2)
        lbg += [0.0]
        ubg += [1.0]

        # update state for next step
        Vs = Vs_after
        up = uk

    G = ca.vertcat(*g)# vertically concatenate all CasADi expressions in the list g
    assert G.is_column()
    assert G.numel() == len(lbg) == len(ubg), "Constraint/bounds length mismatch"
    # after building G, before return

    lbu = np.array([0.0, 0.0], dtype=float)
    ubu = np.asarray(u_max, float)
    # Tile across horizon
    lbx = np.tile(lbu, N)
    ubx = np.tile(ubu, N)

    """ After one single vector G(U)∈R^m that contains all constraints
    if g = [MX(2x1), MX(2x1), MX(2x1)],
    then ca.vertcat(*g) becomes one big MX(6x1).
    """

    #  pack params
    p = ca.vertcat(x0, u_prev0, zsc, uscale,Vs0,V_sens,V_max,C_c) # POPT expects one parameter vector p.

    # --- solver ---
    # x: decision variables (what IPOPT changes),
    # p: parameters (what we provide at solve time),
    # f: objective (the scalar cost to minimize),
    # g: constraint vector (with bounds lbg, ubg).
    nlp = {'x': V, 'p': p, 'f': J, 'g': G}

    solver = ca.nlpsol('solver', 'ipopt', nlp, {
        # Turn OFF lam_p (post-processing that often NaNs with parameters)
        'calc_lam_p': False,          # <-- add this

        # (Optional) keep lam_x if you use it; otherwise you can disable too
        # 'calc_lam_x': True,

        'ipopt': {
            'print_level': 1,
            'sb': 'yes',
            'print_user_options': 'yes',
            'linear_solver': 'mumps',
        },
        'print_time': True
    })
    
    
    
    def pack_p(x0_val, u_prev_val, z_scale_val, u_scale_val,Vs0_val,V_sens_val,V_max_val,C_c_val):
        return np.concatenate([
            np.asarray(x0_val, float).ravel(),
            np.asarray(u_prev_val, float).ravel(),
            np.asarray(z_scale_val, float).ravel(),
            np.asarray(u_scale_val, float).ravel(),
            np.asarray(Vs0_val, float).ravel(),
            np.asarray(V_sens_val, float).ravel(),
            np.asarray(V_max_val, float).ravel(),
            np.asarray(C_c_val, float).ravel(),
            #np.asarray(taus_val, float).ravel(),
        ])

    def unpack_u(Uvec):
        Uvec = np.asarray(Uvec, float).ravel()
        return [Uvec[k*NU:(k+1)*NU] for k in range(N)]

    return solver, pack_p, unpack_u, lbx, ubx, np.array(lbg, float), np.array(ubg, float)



