# PFASProject
![PFAS control unit for flow UV-ARP degradtion reactor](https://github.com/Crelle490/PFASProject/blob/main/media/control_unit.png)

Comprehensive controls, modeling, and simulation code for per- and polyfluoroalkyl substances (PFAS) photochemical degradation. The repository combines a physics-informed neural network (PINN) predictor, an extended Kalman filter (EKF), and a model predictive control (MPC) stack plus hardware drivers for pumps and sensors.

## Repository layout
- `config/` – YAML configuration for controller hardware, initial conditions, kinetic parameters, and noise covariances.
- `predictor/` – Hybrid PINN + EKF state estimator (see `HPINN_predictor.py`) with supporting Jacobian, covariance, and ODE helpers.
- `Controller/` – MPC utilities, live plotting, and a full photodegradation simulation script (`simulate_system.py`).
- `PFAS_CTRL/` – Command-line tools and serial drivers for pumps, pH, and fluoride sensors (`PFAS_CTRL/cli.py`).
- `Models_*` – Trained TensorFlow model code and checkpoints used by the predictor.
- `tests/` – Hardware driver checks and controller simulations.
- `data/` – Example/simulated datasets written by the predictor utilities.

## Prerequisites
- Python 3.10+ recommended.
- Core Python dependencies used across the project include:
  - Numerical/ML stack: `tensorflow`, `numpy`, `pandas`, `matplotlib`, `casadi`.
  - YAML/CLI/tooling: `pyyaml`, `typer`, `serial` (for hardware), and `scipy`.
- Create and activate a virtual environment, then install packages, for example:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install tensorflow numpy pandas matplotlib casadi pyyaml typer pyserial scipy
  ```

Some functionality (e.g., pump/sensor drivers) requires access to serial devices; ensure the user running the code has permission to open the configured ports.

## Configuration
Key runtime parameters live in `config/`:

- `ctrl_config.yaml` configures the pump serial connection for the CLI.
- `physichal_paramters.yaml`, `initial_conditions.yaml`, and `trained_params.yaml` provide kinetic constants, initial concentrations, and trained reaction rates consumed by `HPINN_predictor.HPINNPredictor`.
- `covariance_params.yaml` sets process/measurement noise and initial error covariances used by the EKF.

Edit these files (or point to alternatives via environment variables where applicable) before running simulations or hardware commands.

## Running the predictor
The hybrid predictor loads trained PINN weights and steps the system forward while updating with EKF corrections.

- Generate synthetic fluoride concentration data on the predictor's internal grid:

  ```bash
  python - <<'PY'
  from predictor.HPINN_predictor import HPINNPredictor
  predictor = HPINNPredictor(dt=1.0, sensor_frequency=0.01)
  predictor.simulate_data(save_path="data/simulated_F_concentraction.csv")
  PY
  ```

- Run the sample workflow that injects noisy sensor readings and compares EKF-corrected trajectories against pure PINN predictions:

  ```bash
  python simulated_predictor.py
  ```

## Running the MPC simulation
`Controller/simulate_system.py` stitches together the kinetic integrator, MPC optimizer, EKF, and plotting utilities to simulate PFAS degradation with and without catalyst dosing. Execute it directly to produce plots and degradation time summaries:

```bash
python Controller/simulate_system.py
```

The script reads parameters from `config/`, builds CasADi optimization problems, and produces live/high-resolution trajectories for each PFAS species plus fluoride. Adjust sampling time, batch count, or volume constraints inside the script if experimenting with different setups.

## Pump control CLI
`PFAS_CTRL/cli.py` exposes Typer commands for interacting with a WX10 pump. By default it loads `config/ctrl_config.yaml`, but you can override via `--config` or the `CTRL_CONFIG` environment variable.

Examples:

```bash
python -m PFAS_CTRL.cli state
python -m PFAS_CTRL.cli speed 50 --cw --config config/ctrl_config.yaml
python -m PFAS_CTRL.cli stop
```

## Testing
The `tests/` directory includes controller and hardware driver checks. Many tests expect connected serial devices (pump, flow sensor, pH analyzer, fluoride analyzer, GPIO board), so they may fail without lab hardware. To run the suite when hardware is available:

```bash
pytest
```

## Troubleshooting tips
- If TensorFlow fails to find GPU support, fall back to CPU by ensuring no conflicting CUDA libraries are on the path.
- Serial timeouts when using the CLI typically mean the configured `port`, `address`, or `parity` in `ctrl_config.yaml` does not match the attached hardware.
- When modifying kinetic parameters or covariances, regenerate simulated data and re-run controller simulations to verify stability.
