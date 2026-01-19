"""
Plot the actuation cost (as defined in sweep_costs) when ΣPFAS first reaches 1% and 10%.

Usage: python Controller/plot_cost_time_tradeoff.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import scienceplots

from helper_functions import (
    find_project_root,
    load_yaml_params,
    load_yaml_constants,
    DEFAULT_WEIGHTS,
    make_normalizers_from_numpy,
    advance_one_control_step,
    build_mpc_adi,
    mpc_adi,
    vol_from_deltaC_safe,
    estimate_e_with_intensity
)


# Import integrator without breaking relative paths
try:
    from E_TF_MultipleBatch_Adaptive_c.integrator import RungeKuttaIntegratorCell
except Exception:
    here = Path(__file__).resolve().parent
    model_dir = (here / ".." / "Models_Multiple_Scripts" / "E_TF_MultipleBatch_Adaptive_c").resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from integrator import RungeKuttaIntegratorCell


def run_sim(qx_val, qf_val, steps=50):
    """Run a catalyst-on simulation with custom (qx, qf)."""
    here = Path(__file__).resolve().parent
    root = find_project_root(here)
    cfg_dir = root / "config"
    params, init_vals = load_yaml_params(cfg_dir)
    k_values = load_yaml_constants(cfg_dir)

    pH = float(init_vals["pH"])
    c_cl_0 = float(init_vals["c_cl_0"])
    c_so3_0 = float(init_vals["c_so3_0"])
    intensity_0 = float(init_vals["Intensity"])
    k_list = [k_values[f"k{i}"] for i in range(1, 8)]
    k1, k2, k3, k4, k5, k6, k7 = k_list

    initial_state = np.array([init_vals["c_pfas_init"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_state = initial_state.reshape((1, 1, 8)).astype(np.float32)

    c_cl = params["c_cl"]
    c_so3 = params["c_so3"]
    C_c = [c_so3, c_cl, pH, intensity_0]

    cl_max = c_cl * 0.1
    so3_max = c_so3 * 0.1
    intensity_max = 1.0
    pH_max = 14.0
    u_max = [so3_max, cl_max, pH_max, intensity_max]

    e_max = estimate_e_with_intensity(
        params,
        c_so3=so3_max,
        c_cl=cl_max,
        pH=pH_max,
        c_pfas_init=init_vals["c_pfas_init"],
        k1=k1,
        intensity=intensity_max,
    )
    k_max = max([k1, k2, k3, k4, k5, k6, k7])
    Ts = 50  # int(1.0 / (k_max * e_max))
    dt_sim = Ts / 10.0

    x0_flat = initial_state.reshape(-1)
    weights = {**DEFAULT_WEIGHTS, "qx": float(qx_val), "qf": float(qf_val)}
    R = np.asarray(weights["R"], dtype=float)
    power_uv_lamp = 14
    price_of_electricity = 2.60 # DKK/kWh
    R[3] = power_uv_lamp * Ts * price_of_electricity /(1000*3600)
    Rd = np.asarray(weights["Rd"], dtype=float)
    z_scale, u_scale = make_normalizers_from_numpy(x0_flat, np.asarray(u_max, float))

    Vi = init_vals["Vi"]
    Vmax = init_vals["Vmax"]
    V_sens = init_vals["V_sens"]

    rk_cell = RungeKuttaIntegratorCell(
        k1, k2, k3, k4, k5, k6, k7,
        params, c_cl_0, c_so3_0, pH, intensity_0, dt_sim,
        initial_state.reshape(1, 8), for_prediction=False
    )
    rk_cell.build(input_shape=initial_state.shape)

    substeps = int(round(Ts / dt_sim))
    ctx_adi = build_mpc_adi(
        params=params,
        k_list=k_list,
        c_pfas_init=init_vals["c_pfas_init"],
        dt=dt_sim,
        substeps=substeps,
        N=6,
        weights=weights,
        u_max=u_max,
        x0_flat=x0_flat,
        enable_volume_constraints=True,
        du_max=None,
        rk_cell=rk_cell,
    )

    # histories
    all_states = [x0_flat]
    all_times = [0.0]
    u_hist = []

    uk_prev = np.array([0.0, 0.0, pH, intensity_0], dtype=float)
    for step in range(steps):
        current_state = all_states[-1].copy()
        t_k = all_times[-1]

        uk, Uplan, Jstar = mpc_adi(
            xk_flat=current_state,
            uk_prev=uk_prev,
            ctx=ctx_adi,
            Vs0=Vi,
            V_sens=V_sens,
            V_max=Vmax,
            C_c=C_c,
            warm_start=None,
        )
        u_hist.append(uk.copy())

        # dilution map
        deltau = uk - uk_prev
        deltaC = deltau[0:2]
        Vs_before = Vi - V_sens
        Vsum = 0.0
        for i in range(2):
            Vsum += vol_from_deltaC_safe(deltaC[i], C_c[i], Vs_before, eps=1e-12)
        gamma = 1.0 - Vsum / (Vs_before + Vsum) if (Vs_before + Vsum) > 0 else 1.0
        gamma = float(np.clip(gamma, 0.0, 1.0))
        Vi = float(Vs_before + Vsum)
        current_state = gamma * current_state

        # integrate plant for one interval
        segment_state = current_state.copy()
        for _ in range(substeps):
            y_tf = advance_one_control_step(
                rk_cell,
                segment_state.reshape(1, 1, 8).astype(np.float32),
                uk,
                1,
            )
            segment_state = np.reshape(y_tf[0].numpy(), (-1,))
            segment_state = np.maximum(segment_state, 0.0)
        all_states.append(segment_state.copy())
        all_times.append(t_k + Ts)
        uk_prev = uk

    all_states = np.array(all_states)
    all_times = np.array(all_times)
    u_hist = np.array(u_hist)

    # Compute custom actuation-only cost per step:
    #  - magnitude on u0,u1,u3 (normalized)
    #  - pH (u2) only penalized when it changes
    custom_costs = []
    prev_u = np.array([0.0, 0.0, pH, intensity_0], dtype=float)
    for u in u_hist:
        cost_mag = R[0] * u[0] + R[1] * u[1] + R[3] * u[3]
        du_norm = (u - prev_u)
        cost_pH_change =  Rd[2] * du_norm[2] if Rd.size >= 3 else 0.0
        custom_costs.append((cost_mag + cost_pH_change)*1/9.6e-7)
        prev_u = u

    return all_times, all_states, np.array(custom_costs), u_hist


def main():
    qx_list = [50,75, 100, 150, 200,250]
    qf_list = [50]

    step_arr = [22,17]

    results = {}
    for qx in qx_list: 
        for qf in qf_list:
            n_steps = step_arr[1] if qx >= 100 else step_arr[0]
            print(f"Running qx={qx}, qf={qf} ...")
            t, X, costs, u_hist = run_sim(qx, qf, steps=n_steps)
            results[(qx, qf)] = (t, X, costs, u_hist)

    # Collect cost and time at 1% and 10% of initial ΣPFAS
    tradeoff_points = {0.01: [], 0.10: []}
    for (qx, qf), (t, X, costs, _) in results.items():
        Z = np.sum(X[:, :7], axis=1)
        initial_conc = Z[0]
        for target_frac in (0.01, 0.10):
            target_conc = target_frac * initial_conc
            indices_below_target = np.where(Z <= target_conc)[0]
            if len(indices_below_target) > 0:
                step_to_target = indices_below_target[0]
                time_to_target = t[step_to_target]
                total_cost = costs[step_to_target]*0.13  # cost value to use at the target step
                tradeoff_points[target_frac].append((time_to_target, total_cost))
                print(
                    f"{target_frac*100:.0f}% -> reaches at step {step_to_target}, "
                    f"time {time_to_target:.2f}s, total cost {total_cost:.4e}"
                )
            else:
                print(f"{target_frac*100:.0f}% -> not reached within simulation.")

    # Plot cost at target vs time to target (connect points for indication)
    plt.style.use(["science", "grid"])
    plt.rcParams.update(
        {
            "legend.loc": "best",
            "legend.frameon": False,
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.1,
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
            "text.usetex": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    markers = {0.01: "o", 0.10: "s"}
    colors = {0.01: "#2b6cb0", 0.10: "#c05621"}
    for target_frac, points in tradeoff_points.items():
        if not points:
            continue
        # sort by time to connect points sensibly
        points_sorted = sorted(points, key=lambda p: p[0])
        times = [p[0] for p in points_sorted]
        costs_at_target = [p[1] for p in points_sorted]
        ax.plot(
            times,
            costs_at_target,
            marker=markers.get(target_frac, "o"),
            linestyle="-",
            linewidth=2.0,
            markersize=6.5,
            color=colors.get(target_frac, None),
            label=f"{int(target_frac*100)}% ΣPFAS",
        )

    ax.set_xlabel("Time to target ΣPFAS [s]")
    ax.set_ylabel("Actuation cost at target [EUR/mol] (414.05g PFOA)")
    ax.set_title("Cost vs time at 1% and 10% ΣPFAS")
    ax.grid(True, which="both")
    ax.minorticks_on()
    ax.tick_params(direction="in", length=4, width=1.1)
    ax.tick_params(axis="both", which="minor", length=2.5, width=1.0)
    ax.legend()

    out_path = Path(__file__).resolve().parent.parent / "results" / "cost_vs_time_total_cost.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
