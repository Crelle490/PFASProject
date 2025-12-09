from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helper_functions import (
    find_project_root,
    load_yaml_params,
    load_yaml_constants,
    DEFAULT_WEIGHTS,
)
from casadi_mpc import make_normalizers_from_numpy


def stage_cost_states_only(x_vec: np.ndarray, z_scale: np.ndarray, qx: float) -> float:
    """Compute the PFAS-only part of the stage cost (no input penalties)."""
    x_vec = np.asarray(x_vec, float).reshape(-1)
    z_scale = np.asarray(z_scale, float).reshape(-1)
    x7_norm = x_vec[:7] / (z_scale[:7] + 1e-30)
    return float(qx * np.sum(x7_norm ** 2))


def main():
    # Locate config and load initial values
    root = find_project_root(Path(__file__).resolve().parent)
    cfg_dir = root / "config"
    params, init_vals = load_yaml_params(cfg_dir)
    _ = load_yaml_constants(cfg_dir)  # not used directly here

    # Build normalizers
    c_pfas_init = float(init_vals["c_pfas_init"])
    x0_flat = np.array([c_pfas_init] + [0.0] * 7, dtype=float)
    u_max = np.array(
        [params["c_so3"] * 0.1, params["c_cl"] * 0.1, 14.0, 1.0], dtype=float
    )
    z_scale, _ = make_normalizers_from_numpy(x0_flat, u_max)

    qx = float(DEFAULT_WEIGHTS["qx"])
    species_labels = [
        "PFAS1",
        "PFAS2",
        "PFAS3",
        "PFAS4",
        "PFAS5",
        "PFAS6",
        "PFAS7",
    ]

    # Explore costs over an absolute concentration grid (same for all species)
    max_conc = c_pfas_init * 5.0
    conc_grid = np.linspace(0.0, max_conc, 300)

    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=False)
    axes = axes.flatten()

    for i, label in enumerate(species_labels):
        costs = []
        for c in conc_grid:
            x_vec = np.zeros(8, dtype=float)
            x_vec[i] = c
            costs.append(stage_cost_states_only(x_vec, z_scale, qx))
        ax = axes[i]
        ax.plot(conc_grid, costs, lw=2)
        ax.set_title(f"{label} cost vs concentration")
        ax.set_xlabel("Concentration [M]")
        ax.set_ylabel("Cost contribution")
        ax.grid(True)

    # Hide unused subplot (8th slot)
    axes[-1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
