import numpy as np
import tensorflow as tf
import yaml

from tensorflow.keras.layers import RNN
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import custom modules for PINN
from .HPINN_Fused_model import RungeKuttaIntegratorCell, PINNModel, interpolate_predictions 

# Import custom modules for EKF
from predictor.EKF import ExtendedKalmanFilter
from predictor.kinetic_model import f, h

class HPINNPredictor:
    def __init__(self,dt,sensor_frequency):
        # Load parameters from config folder
        with open("./config/physichal_paramters.yaml", "r") as file:
            self.params = yaml.safe_load(file)

        with open("./config/initial_conditions.yaml", "r") as file:
            init_vals = yaml.safe_load(file)

        with open("./config/trained_params.yaml", "r") as file:
            trained_reaction_rates = yaml.safe_load(file)

        with open("./config/covariance_params.yaml", "r") as file:
            cov_params = yaml.safe_load(file)

        # Assign initial conditions
        self.pH = init_vals["pH"]
        self.c_pfas_init = init_vals["c_pfas_init"]
        self.c_cl = init_vals["c_cl"]
        self.c_so3 = init_vals["c_so3"]

        # Trained reaction rate constants (these could come from a file or be hard-coded)
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7 = [trained_reaction_rates[f'k{i}'] for i in range(1, 8)]

        # Assembel reaction rate constant in array for jacobian
        self.k = [self.k1,self.k2,self.k3,self.k4,self.k5,self.k6,self.k7]

        # Assign covarinace paameters from configuration file
        self.Q = np.array(cov_params["process_noise_covariance"])
        self.R = np.array(cov_params["measurement_noise_covariance"])
        self.P0 = np.array(cov_params["initial_error_covariance"])

        # Set simulation resolution
        self.dt_sim = dt
        self.freqeuncy = sensor_frequency
        self.period = 1/self.freqeuncy
        self.N_sim = int(self.period/self.dt_sim)
        self.current_time = 0

        # Initial state for shape inference
        self.init_state = np.zeros((1, 8), dtype=np.float32)
        self.init_state[0, 0] = self.c_pfas_init  # Set PFAS initial concentration
        self.concentractions = self.init_state

        # Set initial input value
        self.u = 0

        # Initialize simulation time
        self.set_simulation_time()

        # Initialize simulation time
        self.set_initial_estimation_state()

        # Initialize PINN
        self._build_HPINN()

        # Determine the concentraction of electrons
        self.calculate_eqa()

        # Initialize EKF
        self._build_EKF()
    
    def calculate_eqa(self):
        # Calculate hydrated electron generation.
        numerator = self.rk_cell.generation_of_eaq()

        # Additional kinetic parameters.
        k_so3_eaq = 1.5e6
        k_cl_eaq = 1e6
        beta_j = 2.57e4

        denominator = self.k1 * self.c_pfas_init + beta_j + k_so3_eaq * self.rk_cell.c_so3 + k_cl_eaq * self.rk_cell.c_cl

        # Hydrated electron concentration.
        self.c_eaq = numerator / denominator
    
    def set_simulation_time(self):
        simulation_time = np.arange(self.N_sim) * self.dt_sim + self.current_time
        simulation_time = simulation_time.reshape(1, -1, 1)
        self.simulation_time = tf.convert_to_tensor(simulation_time, dtype=tf.float32)

    def set_initial_estimation_state(self):
        # Concentractions must be (1,8)
        # Assign initial state for HPINN
        try:
            self.initial_state = tf.convert_to_tensor(self.concentractions, dtype=tf.float32)
            if self.initial_state.shape != (1, 8):
                raise ValueError(f"initial_state shape is {self.initial_state.shape}, expected (1, 8)")
        except Exception as e:
            print(f"Error setting initial_state: {e}")
            raise  # or handle as needed
    
    def predict_state(self):
        estimation_input = [self.simulation_time, self.initial_state]
        y_pred = self.model.predict(estimation_input)
        for i in range(0,self.N_sim):
            self.ekf.predict(y_pred[0,i,:],u=self.u)
        return y_pred
    
    def update_input(self):
        # If the addetives change the concentraction of eaq this function can be updated
        self.u = 0
    
    def get_sensor_measuerment(self,z):
        self.sensor_measuerment = z
    
    def step(self):
        self.set_initial_estimation_state()
        PINN_prediction = self.predict_state()
        self.ekf.update(self.sensor_measuerment)
        x = self.ekf.x  # Shape (8,)
        self.concentractions = [x]  # Extract PFAS concentrations
        # Update time
        self.current_time = self.simulation_time[:,-1].numpy()
        self.set_simulation_time()

        return PINN_prediction, self.concentractions, self.sensor_measuerment
    
    def update_models(self):
        """
        Update freqeuncy and dt to change time scale of the HPINN model.
        This is primarily used for evaluation and testing purposes.
        """
        # Set simulation resolution
        self.period = 1/self.freqeuncy
        self.N_sim = int(self.period/self.dt_sim)
        self.current_time = 0

        # Initialize simulation time
        self.set_simulation_time()

        # Initialize simulation time
        self.set_initial_estimation_state()

        # Initialize PINN
        self._build_HPINN()

        # Determine the concentraction of electrons
        self.calculate_eqa()

        # Initialize EKF
        self._build_EKF()

    def simulate_data(self, steps=None, start_at_zero=True,
                  save_path="data/simulated_F_concentraction.csv"):
        """
        Run the current HPINN model forward and save the F- (fluoride) series.
        Used to generate synthetic data for testing and evaluation. 

        Parameters
        ----------
        steps : int or None
            Number of time steps to simulate. If None, uses self.N_sim.
        start_at_zero : bool
            If True, simulate from t=0..(steps-1)*dt.
            If False, simulate continuing from current_time.
        save_path : str
            Where to save the CSV. Default: 'data/simulated_F_concentraction.csv'
        """
        # ----- build time grid -----
        if steps is None:
            steps = int(self.N_sim)

        if start_at_zero:
            t = np.arange(steps, dtype=np.float32) * float(self.dt_sim)
        else:
            # continue from current time
            t0 = float(np.array(self.current_time).reshape(-1)[-1]) if hasattr(self, "current_time") else 0.0
            t = (np.arange(steps, dtype=np.float32) * float(self.dt_sim)) + t0

        t_tensor = tf.convert_to_tensor(t.reshape(1, -1, 1), dtype=tf.float32)

        # ----- run model -----
        # Use the predictor's current initial_state so c_eaq and dynamics are consistent
        estimation_input = [t_tensor, self.initial_state]
        y_pred = self.model.predict(estimation_input, verbose=0)  # (1, steps, 8)
        F = y_pred[0, :, -1]  # last column is fluoride

        # ----- save -----
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, F, delimiter=",", fmt="%.10f")

    def _build_HPINN(self):
        """
        Build the HPINN model using the Runge-Kutta integrator cell and PINNModel.
        This method initializes the model and loads the trained weights.
        """
        # Build integrator cell and model
        self.rk_cell = RungeKuttaIntegratorCell(self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7, self.params, self.c_cl, self.c_so3, self.pH, self.dt_sim, self.init_state,for_prediction=True)
        self.model = PINNModel(self.rk_cell, num_steps=self.N_sim)  # Adjust num_steps based on desired simulation time
        # Build model (required before loading weights)
        dummy_input = np.zeros((1, self.N_sim, 1), dtype=np.float32)     # Dummy input must have shape (batch, T_sim_max, 1)
        dummy_input = tf.convert_to_tensor(dummy_input, dtype=tf.float32)

        _ = self.model([dummy_input, self.initial_state])

        # Load trained weights
        self.model.load_weights("./checkpoints/pinn_model.weights.h5")
    
    def _build_EKF(self):
        """
        Build the Extended Kalman Filter (EKF).
        """

        # Assign initial state
        x0 = np.array([self.c_pfas_init] + [0.0]*7)  # shape (1,8)

        self.x0 = x0
        self.ekf = ExtendedKalmanFilter(
            Q=self.Q,
            R=self.R,
            x0=x0,
            P0=self.P0,
            k=self.k,
            c_eaq=self.c_eaq,
            dt=self.dt_sim,
        )

