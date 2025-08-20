
from HPINN_predictor import HPINNPredictor
import pandas as pd
import numpy as np
from predictor.kinetic_model import f 
import matplotlib.pyplot as plt
import tensorflow as tf

# Prediction horeizon defined as number of steps to simulat response. Total prediction time would be N_sim*dt_sim
dt = 1
N = 3601
frequency = 1/N

# Setup predictor
predictor = HPINNPredictor(dt=dt,sensor_frequency=frequency)
predictor.simulate_data(steps=N)