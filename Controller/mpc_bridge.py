# mpc_bridge.py

"""
Use MPC by physical hardware loop
"""

from __future__ import annotations
import numpy as np
from functools import lru_cache

from helper_functions import (
    find_project_root, load_yaml_params, load_yaml_constants,
    make_normalizers_from_numpy, build_mpc_adi, mpc_adi
)

@lru_cache(maxsize=2)
def _cached_ctx(Ts: float, N: int, u_max_tuple: tuple, x0_tuple: tuple):
    """
    Build + cache the CasADi MPC context. Cache key uses (Ts, N, u_max, x0).
    Rebuilds only when those change.
    """
    # project + config
    import pathlib as _pl
    root = find_project_root(_pl.Path(__file__).resolve())
    cfg_dir = root / "config"

    # params
    params, init_vals = load_yaml_params(cfg_dir)
    k_vals = load_yaml_constants(cfg_dir)
    k = [k_vals[f'k{i}'] for i in range(1, 8)]
    pH = float(init_vals["pH"])
    c_pfas_init = float(init_vals["c_pfas_init"])

    # normalizers
    x0_flat = np.array(x0_tuple, dtype=float)
    u_max = np.array(u_max_tuple, dtype=float)
    z_scale, u_scale = make_normalizers_from_numpy(x0_flat, u_max)

    # controller discretization
    dt_sim = 1.0
    substeps = int(round(Ts / dt_sim))

    # weights (import from your helper or inline)
    from helper_functions import DEFAULT_WEIGHTS

    ctx = build_mpc_adi(
        params=params,
        k_list=k,
        pH=pH,
        c_pfas_init=c_pfas_init,
        dt=dt_sim,
        substeps=substeps,
        N=int(N),
        weights=DEFAULT_WEIGHTS,
        u_max=u_max,
        x0_flat=x0_flat,
        enable_volume_constraints=True,
        du_max=None,
        rk_cell=None,  # not needed to *solve*
    )
    # expose z/u scales if you want them downstream
    ctx["z_scale"] = z_scale
    ctx["u_scale"] = u_scale
    return ctx

def mpc_compute_u_once(
    xk_flat: np.ndarray,            # shape (8,)
    uk_prev: np.ndarray,            # shape (2,)
    *,
    Ts: float,
    N: int,
    u_max: np.ndarray,              # (2,) [so3_max, cl_max]
    Vi: float,                      # current volume [L]
    V_sens: float,                  # [L] sampled each step
    V_max: float,                   # [L] max vessel volume
    C_c: np.ndarray | list,         # (2,) stock conc [M]
    hysteresis=(5e-10, 2e-10, 2),   # (z_start, z_stop, H)
):
    """
    Returns (u_k, U_plan, J_star), using your existing mpc_adi().
    This is safe to import and call from *any* folder.
    """
    xk_flat = np.asarray(xk_flat, float).ravel()
    uk_prev = np.asarray(uk_prev, float).ravel()
    u_max   = np.asarray(u_max, float).ravel()
    C_c     = np.asarray(C_c, float).ravel()

    ctx = _cached_ctx(
        Ts=float(Ts),
        N=int(N),
        u_max_tuple=tuple(u_max.tolist()),
        x0_tuple=tuple(xk_flat.tolist()),
    )

    # simple hysteresis like in your simulate()
    z = float(np.sum(xk_flat[:7]))
    z_start, z_stop, H = hysteresis

    # You can pass history in if you like; for now keep the MPC always active:
    if False and z < z_stop:
        u_k = uk_prev
        U_plan = np.repeat(uk_prev[None, :], ctx["N"], axis=0)
        J_star = 0.0
    else:
        u_k, U_plan, J_star = mpc_adi(
            xk_flat=xk_flat, uk_prev=uk_prev, ctx=ctx,
            Vs0=Vi, V_sens=V_sens, V_max=V_max, C_c=C_c, warm_start=None
        )

    return np.asarray(u_k, float), np.asarray(U_plan, float), float(J_star)
