"""
Run multiple qx/qf sweeps, save results, and plot overlay of ΣPFAS and cost.
Uses simulate_system helpers; data is saved so plots can be redrawn without rerunning.
"""

from pathlib import Path
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# Import simulation helpers (no heavy sim runs thanks to main-guard in simulate_system)
from simulate_system import (
    simulate,
    build_ctx_for_weights,
    DEFAULT_WEIGHTS,
    Ts,
    Vi,
)


def default_weight_grid():
    """Build a richer grid of qx/qf pairs around the current defaults."""
    base_ctx = build_ctx_for_weights()
    base_qx = base_ctx["weights_cfg"]["qx"]
    base_qf = base_ctx["weights_cfg"]["qf"]

    scale_pairs = [
        (0.5, 0.5),
        (0.75, 1.0),
        (1.0, 1.0),
        (1.25, 1.25),
        (1.5, 1.0),
        (1.0, 1.5),
        (2.0, 2.0),
    ]

    grid = []
    for sx, sf in scale_pairs:
        grid.append(
            {
                "label": f"qx{sx:.2g}_qf{sf:.2g}",
                "qx": sx * base_qx,
                "qf": sf * base_qf,
            }
        )
    return grid


def run_weight_sweep(weight_configs, steps, Vi):
    """
    Run a set of MPC simulations for different qx/qf pairs.
    weight_configs: list of {"qx": float, "qf": float, "label": str}
    """
    results = []
    for cfg in weight_configs:
        overrides = {k: cfg[k] for k in ("qx", "qf") if k in cfg}
        label = cfg.get("label") or f"qx={overrides.get('qx')}, qf={overrides.get('qf')}"
        ctx = build_ctx_for_weights(overrides)
        sim_out = simulate(
            with_catalyst=True,
            steps=steps,
            Vi=Vi,
            ctx=ctx,
            weight_label=label,
            enable_live_plot=False,
        )

        results.append(
            {
                "label": label,
                "qx": overrides.get("qx", DEFAULT_WEIGHTS["qx"]),
                "qf": overrides.get("qf", DEFAULT_WEIGHTS["qf"]),
                "t_plot": sim_out[4],
                "X_plot": sim_out[3],
                "cost_trace": sim_out[-1],
            }
        )
    return results


def plot_runs_side_by_side(runs, Ts, save_path=None):
    """
    Draw ΣPFAS (states) and cost traces in one figure (two stacked subplots).
    """
    if not runs:
        return None

    fig, (ax_state, ax_cost) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for res in runs:
        total_pfas = np.sum(res["X_plot"][:, :7], axis=1)
        label = f"{res['label']} (qx={res['qx']:.3g}, qf={res['qf']:.3g})"
        ax_state.plot(res["t_plot"], total_pfas, label=label, linewidth=2.0)

        t_cost = np.arange(len(res["cost_trace"])) * Ts
        ax_cost.plot(t_cost, res["cost_trace"], label=label, linewidth=2.0)

    ax_state.set_ylabel(r"$\Sigma$ PFAS [M]", fontsize=13, fontweight="bold")
    ax_state.set_title("State evolution vs. qx/qf", fontsize=14, fontweight="bold")
    ax_state.grid(True)
    ax_state.legend(fontsize=10)

    ax_cost.set_xlabel("Time [s]", fontsize=13, fontweight="bold")
    ax_cost.set_ylabel("Cost J*", fontsize=13, fontweight="bold")
    ax_cost.set_title("MPC objective vs. qx/qf", fontsize=14, fontweight="bold")
    ax_cost.grid(True)
    ax_cost.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def save_runs(runs, Ts, path):
    """Persist simulation outputs so plots can be regenerated without reruns."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"Ts": Ts, "runs": runs}
    np.save(path, payload, allow_pickle=True)
    return path


def load_runs(path):
    data = np.load(path, allow_pickle=True).item()
    return data["runs"], data["Ts"]


def main():
    steps = 15
    weight_configs = default_weight_grid()

    # Run sweep (live plots disabled inside simulate)
    runs = run_weight_sweep(weight_configs, steps=steps, Vi=Vi)

    # Save raw data for later plotting
    data_path = Path("results/cost_exploration.npy")
    save_runs(runs, Ts, data_path)
    print(f"Saved sweep data to {data_path}")

    # Plot overlays (states + cost)
    fig_path = Path("results/cost_exploration_overlay.png")
    plot_runs_side_by_side(runs, Ts, save_path=fig_path)
    print(f"Saved overlay plot to {fig_path}")

    plt.show()


if __name__ == "__main__":
    main()
