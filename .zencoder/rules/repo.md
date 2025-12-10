---
description: Repository Information Overview
alwaysApply: true
---

# PFASProject Information

## Summary
Physics-Informed Neural Network (PINN) based project for PFAS degradation modeling and control. Implements Model Predictive Control (MPC) using CasADi for UV/peroxide treatment systems, with State Estimation via Extended Kalman Filter (EKF). Includes hardware drivers for pumps, pH/fluoride sensors, and flow controllers.

## Structure
- **PFAS_CTRL/**: Main package with system control and hardware drivers (pump WX10, GPIO, flow sensors, pH/fluoride analyzers)
- **Controller/**: MPC implementation using CasADi symbolic math, simulation and live plotting
- **Models_Single_Scripts/ & Models_Multiple_Scripts/**: PINN model implementations (TensorFlow and PyTorch)
- **predictor/**: State estimation with HPINN, EKF, Jacobian computation, and covariance building
- **config/**: YAML configuration files for parameters, initial conditions, and control settings
- **tests/**: Test suite for system and drivers
- **data/**: Sample CSV data files

## Language & Runtime
**Language**: Python  
**Version**: 3.12.3  
**Virtual Environment**: `.venv/` (already configured)  
**Package Manager**: pip

## Dependencies

**Core Libraries**:
- **tensorflow**: Deep learning and PINN model implementation
- **casadi**: Symbolic math and optimization (MPC formulation)
- **numpy**: Numerical computations
- **pandas**: Data handling and CSV operations
- **matplotlib**: Plotting and visualization
- **pyyaml**: Configuration file parsing

**CLI & Control**:
- **typer**: CLI framework for command-line interface
- **pyserial**: Hardware serial communication (sensors/pumps)

**Optional**:
- **pytorch**: Alternative to TensorFlow for PINN models
- **scipy**: Scientific computing utilities

## Build & Installation

```bash
# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # (if available)
# Or install manually:
pip install tensorflow casadi numpy pandas matplotlib pyyaml typer pyserial
```

## Main Entry Points

**CLI Application** (`PFAS_CTRL/cli.py`):
```bash
python -m PFAS_CTRL.cli speed <rpm>      # Set pump speed
python -m PFAS_CTRL.cli stop             # Stop pump
python -m PFAS_CTRL.cli prime            # Prime mode
python -m PFAS_CTRL.cli state            # Read pump state
```

**Configuration**: Uses `config/ctrl_config.yaml` for pump and hardware settings

**Simulations**:
- `Controller/simulate_system.py`: Full system MPC simulation
- `Controller/simulate_full_system.py`: Extended simulation with visualization
- `simulated_predictor.py`: HPINN predictor testing

## Testing

**Location**: `tests/` directory  
**Test Files**:
- `test_system.py`: System controller and integration tests
- `test_drivers.py`: Hardware driver tests
- `data.py`: Data utility tests
- `logger.py` & `logger_test_system.py`: Logging tests

**Run Tests**:
```bash
python -m pytest tests/
# Or directly:
python tests/test_system.py
python tests/test_drivers.py
```

## Configuration

YAML-based configuration in `config/`:
- **ctrl_config.yaml**: Pump serial port, baudrate, and communication settings
- **physichal_paramters.yaml**: Physical constants for degradation kinetics
- **initial_conditions.yaml**: System initial state values
- **trained_params.yaml**: Pre-trained PINN parameters
- **covariance_params.yaml**: EKF covariance matrices

Environment variable for config override:
```bash
export CTRL_CONFIG=/path/to/custom/config.yaml
```

## Model Types

**Single-Batch Models** (`Models_Single_Scripts/`):
- TensorFlow with fixed/adaptive hydrated electron concentration (`c_eaq`)
- PyTorch implementations available

**Multi-Batch Models** (`Models_Multiple_Scripts/`):
- TensorFlow training on multiple sequences
- Integration with RK4 Runge-Kutta solver
- Adaptive parameter tuning

**Predictor** (`predictor/`):
- HPINN predictor with Runge-Kutta integration
- EKF for sensor correction and state estimation
