from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from helper_functions import find_project_root, load_yaml_params, load_yaml_constants
from utils import estimate_e_with_intensity


def main():
    # Load parameters and initial conditions
    root = find_project_root(Path(__file__).resolve().parent)
    cfg_dir = root / "config"
    params, init_vals = load_yaml_params(cfg_dir)
    k_vals = load_yaml_constants(cfg_dir)

    # Base values (held constant when sweeping another input)
    base_so3 = float(init_vals.get("c_so3_0", 0.0))
    base_cl = float(init_vals.get("c_cl_0", 0.0))
    base_pH = float(init_vals.get("pH", 7.0))
    base_intensity = float(init_vals.get("Intensity", 1.0))
    c_pfas_init = float(init_vals["c_pfas_init"])
    k1 = float(k_vals["k1"])

    # Ranges for sweeps (simple linear grids)
    so3_grid = np.linspace(0.0, float(params.get("c_so3as", 0.1)), 200)
    cl_grid = np.linspace(0.0, float(params.get("c_cldas", 0.1)), 200)
    pH_grid = np.linspace(1.0, 14.0, 200)
    intensity_grid = np.linspace(0.0, 1.2 * base_intensity, 200)

    def e_eval(c_so3, c_cl, pH, intensity):
        return estimate_e_with_intensity(
            params,
            c_so3=c_so3,
            c_cl=c_cl,
            pH=pH,
            c_pfas_init=c_pfas_init,
            k1=k1,
            intensity=intensity,
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Sweep SO3
    e_so3 = [e_eval(v, base_cl, base_pH, base_intensity) for v in so3_grid]
    axes[0].plot(so3_grid, e_so3, lw=2)
    axes[0].set_xlabel("SO$_3^{2-}$ [M]")
    axes[0].set_ylabel("e$_{aq}^-$ [M]")
    axes[0].set_title("e vs SO$_3^{2-}$")
    axes[0].grid(True)

    # Sweep Cl
    e_cl = [e_eval(base_so3, v, base_pH, base_intensity) for v in cl_grid]
    axes[1].plot(cl_grid, e_cl, lw=2, color="tab:orange")
    axes[1].set_xlabel("Cl$^-$ [M]")
    axes[1].set_ylabel("e$_{aq}^-$ [M]")
    axes[1].set_title("e vs Cl$^-$")
    axes[1].grid(True)

    # Sweep pH
    e_pH = [e_eval(base_so3, base_cl, v, base_intensity) for v in pH_grid]
    axes[2].plot(pH_grid, e_pH, lw=2, color="tab:green")
    axes[2].set_xlabel("pH")
    axes[2].set_ylabel("e$_{aq}^-$ [M]")
    axes[2].set_title("e vs pH")
    axes[2].grid(True)

    # Sweep intensity
    e_int = [e_eval(base_so3, base_cl, base_pH, v) for v in intensity_grid]
    axes[3].plot(intensity_grid, e_int, lw=2, color="tab:red")
    axes[3].set_xlabel("Intensity (relative)")
    axes[3].set_ylabel("e$_{aq}^-$ [M]")
    axes[3].set_title("e vs intensity")
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
