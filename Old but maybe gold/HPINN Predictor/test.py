# test.py - used to test kinetic model
# 

#### Importing necessary libraries ####
import os, sys
import numpy as np
from pathlib import Path
from ode_runtime import build_model_from_config

here = Path(__file__).resolve().parent          # .../PFASProject/HPINN Predictor
project_root = here.parent                      # .../PFASProject

# so ode_runtime can find create_model:
model_dir = project_root / "Models Multiple Scripts" / "E_TF_MultipleBatch_Adaptive_c"
sys.path.insert(0, str(model_dir.resolve()))



#### Main Function ####
def main():
    # Random time serie
    t = np.arange(0, 601, 1, dtype=np.float32)

    # Build model
    model, dummy, x0 = build_model_from_config(
        cfg_dir=project_root / "config",                      #
        trained_k_yaml=project_root / "config" / "trained_params.yaml",
        t_sim=t,
        dt=1.0
    )

    # Predict based on time series and initial state
    y = model.predict([dummy, x0], verbose=0)
    print("y_pred shape:", y.shape)
    print("first timestep:", y[0, 0])
    print("last timestep:", y[0, -1])

if __name__ == "__main__":
    main()
