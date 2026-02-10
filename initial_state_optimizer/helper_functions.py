import yaml
import numpy as np
import sys
from pathlib import Path

def load_trained_k(path):
    d = yaml.safe_load(open(path, "r"))
    keys = [f"k{i}" for i in range(1, 8)]
    return np.array([d[k] for k in keys], dtype=np.float32)


def load_constants(cfg_dir):
    return yaml.safe_load(open(Path(cfg_dir) / "physichal_paramters.yaml", "r"))


def load_initials(cfg_dir):
    d = yaml.safe_load(open(Path(cfg_dir) / "initial_conditions.yaml", "r"))
    c_cl = d.get("c_cl", d.get("c_cl_0"))
    c_so3 = d.get("c_so3", d.get("c_so3_0"))
    if c_cl is None or c_so3 is None:
        raise KeyError("initial_conditions.yaml must define c_cl/c_so3 or c_cl_0/c_so3_0")
    return float(d["pH"]), float(c_cl), float(c_so3), float(d["c_pfas_init"])