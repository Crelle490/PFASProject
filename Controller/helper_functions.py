from pathlib import Path
import yaml

def find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "config").is_dir():
            return p
    return start

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


def load_yaml_constants(cfg_dir: Path):
    phys_path = cfg_dir / 'trained_params.yaml'
    if phys_path is None:
        raise FileNotFoundError(f"Could not find trained parameters file: {phys_path}")
    with open(phys_path, "r") as f:
        params = yaml.safe_load(f)

    return params