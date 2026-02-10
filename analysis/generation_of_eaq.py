import numpy as np

import sys
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.ode_runtime import build_model_from_config, load_constants

class GenerationOfElectrons():
    def __init__(self,
                 constants, c_so3,pH, dummy_initial_state, **kwargs):
        """
        RK4 integrator cell with adaptive hydrated electron generation.
        dummy_initial_state: (1, 8) array used only to build the cell.
        """
        self.initial_state = np.asarray(dummy_initial_state, dtype=np.float32)

        self.constants = constants

        self.c_cl = 0.0
        self.c_so3 = float(c_so3)
        self.pH = float(pH)

   
    def _fun(self,c_so3):
        self.c_so3 = c_so3
        # Hydrated electron generation (scalar)
        numerator = self.generation_of_eaq()

        # Additional kinetic constants
        k_so3_eaq = 1.5e6
        k_cl_eaq  = 1.0e6
        beta_j    = 2.57e4
        k1 = 2638859520.0

        # Use initial PFAS concentration from dummy initial state (first species)
        c_pfas_init = float(self.initial_state[0, 0])

        denominator = k1 * c_pfas_init + beta_j + k_so3_eaq * self.c_so3 + k_cl_eaq * self.c_cl
        c_eaq = numerator / denominator 

        return c_eaq

    def generation_of_eaq(self):
        """
        Compute the generation rate of hydrated electrons (e_aq−) from 185/254 nm absorption.
        Returns a scalar (float).
        """
        p = self.constants
        c_pfas_init = float(self.initial_state[0, 0])
        
        # [OH-] from pH
        c_oh_m = np.power(10.0, -14.0 + self.pH)

        # Total absorption @185 nm
        Sigma_f_185 = (p["epsilon_h2o_185"] * p["c_h2o"] +
                       p["epsilon_oh_m_185"] * c_oh_m +
                       p["epsilon_cl_185"]   * self.c_cl +
                       p["epsilon_so3_185"]  * self.c_so3 +
                       p["epsilon_pfas_185"] * c_pfas_init)

        # Total absorption @254 nm
        Sigma_f_254 = (p["epsilon_h2o_254"] * p["c_h2o"] +
                       p["epsilon_so3_254"]  * self.c_so3 +
                       p["epsilon_pfas_254"] * c_pfas_init)

        # Fractions @185
        f_h2o_185 = (p["epsilon_h2o_185"] * p["c_h2o"]) / Sigma_f_185
        f_oh_m_185 = (p["epsilon_oh_m_185"] * c_oh_m) / Sigma_f_185
        f_cl_185   = (p["epsilon_cl_185"]   * self.c_cl) / Sigma_f_185
        f_so3_185  = (p["epsilon_so3_185"]  * self.c_so3) / Sigma_f_185

        # Fraction @254
        f_so3_254 = (p["epsilon_so3_254"] * self.c_so3) / Sigma_f_254

        # Contributions @185
        term_h2o_185 = f_h2o_185 * p["phi_h2o_185"] * (1.0 - np.power(10.0, -p["epsilon_h2o_185"] * p["l"] * p["c_h2o"]))
        term_oh_m_185 = f_oh_m_185 * p["phi_oh_m_185"] * (1.0 - np.power(10.0, -p["epsilon_oh_m_185"] * p["l"] * c_oh_m))
        term_cl_185   = f_cl_185   * p["phi_cl_185"]   * (1.0 - np.power(10.0, -p["epsilon_cl_185"]   * p["l"] * self.c_cl))
        term_so3_185  = f_so3_185  * p["phi_so3_185"]  * (1.0 - np.power(10.0, -p["epsilon_so3_185"]  * p["l"] * self.c_so3))
        numerator_185 = p["I0_185"] * (term_h2o_185 + term_oh_m_185 + term_cl_185 + term_so3_185)

        # Contribution @254
        numerator_254 = p["I0_254"] * f_so3_254 * p["phi_so3_254"] * (1.0 - np.power(10.0, -p["epsilon_so3_254"] * p["l"] * self.c_so3))

        return float(numerator_185 + numerator_254)

def load_yaml_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML must parse to a dict, got {type(data)}")

    # Coerce numeric-looking values to float/int (keeps non-numeric strings intact)
    out = {}
    for k, v in data.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, str):
            try:
                # handles "2.07e-6" if it came in as string
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v

    return out

REQUIRED_PHYS_KEYS = [
    "l", "I0_185", "I0_254", "c_h2o",
    "epsilon_h2o_185", "phi_h2o_185",
    "epsilon_h2o_254", "phi_h2o_254",
    "epsilon_oh_m_185", "phi_oh_m_185",
    "epsilon_cl_185", "phi_cl_185",
    "epsilon_so3_185", "phi_so3_185",
    "epsilon_so3_254", "phi_so3_254",
    "epsilon_pfas_185", "epsilon_pfas_254",
]

def validate_keys(d: dict, required: list[str], name="params"):
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"Missing keys in {name}: {missing}")

if __name__ == "__main__":
    cfg_dir = PROJECT_ROOT / "config"
    params_file = PROJECT_ROOT / "config" / "physichal_paramters.yaml"
    constants = load_constants(cfg_dir)

    phys = load_yaml_params(params_file)
    validate_keys(phys, REQUIRED_PHYS_KEYS, name="physichal_paramters.yaml")
    constants = {**constants, **phys}

    initial_states = np.zeros((1, 8), np.float32)
    initial_states[0,0] = 9.074690e-07
    gen = GenerationOfElectrons(constants, c_so3=0,pH=5.7, dummy_initial_state=initial_states)


    c_so3 = np.linspace(0,0.01,100)

    c_eaq = []

    for concentractions in c_so3:
        c_eaq.append(gen._fun(concentractions))

    plt.plot(c_so3,c_eaq)
    plt.show()


    