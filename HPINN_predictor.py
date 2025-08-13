import numpy as np
import tensorflow as tf
import yaml
from model_testing.HPINN_Fused_model import RungeKuttaIntegratorCell, PINNModel, interpolate_predictions 
from tensorflow.keras.layers import RNN
import time
import matplotlib.pyplot as plt
from predictor.jacobian import Jacobian
from predictor.EKF import ExtendedKalmanFilter
from predictor.kinetic_model import f, h
import pandas as pd

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
        print(self.N_sim)
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
            self.ekf.predict(self.u)
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
        x = self.ekf.x  # Shape (9,)
        print(self.ekf.K)
        self.concentractions = [[x[1], x[2], x[3], x[5], x[6], x[7], x[8], x[4]]]  # Extract PFAS concentrations
        # Update time
        self.current_time = self.simulation_time[:,-1].numpy()
        self.set_simulation_time()

        return PINN_prediction, self.concentractions, self.sensor_measuerment
    


    def _build_HPINN(self):
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
        jacobian = Jacobian(self.k)

        # Assign initial state
        x0 = np.array([self.c_eaq, self.c_pfas_init] + [0.0]*7)  # shape (1,9)

        self.ekf = ExtendedKalmanFilter(
            f=f,
            h=h,
            F_jacobian=jacobian.jacobian_reaction,
            H_jacobian=jacobian.jacobian_observation,
            Q=self.Q,
            R=self.R,
            x0=x0,
            P0=self.P0,
            k=self.k,
        )


# Prediction horeizon defined as number of steps to simulat response. Total prediction time would be N_sim*dt_sim
dt = 1
frequency = 0.01
N = int(1/frequency)

# Setup predictor
predictor = HPINNPredictor(dt=dt,sensor_frequency=frequency)

simulated_sensor_data = pd.read_csv('./data/simulated_F_concentraction.csv')

# Convert to NumPy array
data_array = simulated_sensor_data.values  # or simulated_sensor_data.to_numpy()

# Define noise parameters
mean = 0
std_dev = 0.01  # adjust noise level as needed

# Generate Gaussian noise with same shape as data_array
noise = np.random.normal(mean, std_dev, size=data_array.shape)

# Add noise to data
noisy_data = data_array 

# Simulated sensor data used
sensor_data_for_simulation = noisy_data[N+1::N+1]

y_preds = []
t_inputs = []
sensor_measurements = []

for sensor_data in sensor_data_for_simulation:
    predictor.get_sensor_measuerment(z=sensor_data)
    t_input = predictor.simulation_time
    PINN_prediction, concentractions, sensor_measuerment = predictor.step()
    y_pred = PINN_prediction
    print(type(t_input))
    y_pred_reshaped = y_pred.reshape(N, 8)
    t_input_reshaped = t_input.numpy().reshape(N, 1)
    
    y_preds.append(y_pred_reshaped)
    t_inputs.append(t_input_reshaped)
    sensor_measurements.append(sensor_measuerment)

y_pred = np.array(y_preds).reshape(len(sensor_data_for_simulation)*N,8)
print(y_pred.shape)
t_input = np.array(t_inputs).reshape(len(sensor_data_for_simulation)*N,1)
print(t_input.shape)
sensor_measurement = np.array(sensor_measurements)

def plot_prediction(y_pred,t_input):
    labels = ['C7F15COO-','C6F13COO-','F-', 'C5F11COO-','C4F9COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']
    for i, label in enumerate(labels):
        plt.subplot(4, 2, i + 1)
        plt.plot(t_input, y_pred[:,i], 'r', label='Predicted reaction')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        #plt.ylim(-0.015, 0.015)
        plt.grid(True)
        plt.legend()
    plt.show()

plot_prediction(y_pred,t_input)


