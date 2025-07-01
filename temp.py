import numpy as np
#import tensorflow as tf
import yaml
#from HPINN_Fused_model import RungeKuttaIntegratorCell, PINNModel, interpolate_predictions 
#from tensorflow.keras.layers import RNN
import time
import matplotlib.pyplot as plt
from predictor.jacobian import Jacobian
from predictor.EKF import ExtendedKalmanFilter
from predictor.kinetic_model import f, h
import tensorflow as tf
import pandas as pd

df = pd.read_csv('./data/Batch_PFAS_data.csv')  # CSV columns: sequence_id, time (s), C7F15COO-, C5F11COO-, C3F7COO-, C2F5COO-, CF3COO-
groups = df.groupby("sequence_id")
columns = ["C7F15COO-", "C5F11COO-", "C3F7COO-", "C2F5COO-", "CF3COO-"]

initial_states_list = []  # each initial state: (8,)

for seq_id, group in groups:
    group_sorted = group.sort_values(by="time (s)")
    t_seq = group_sorted["time (s)"].values   # shape (T_exp,)
    y_seq = group_sorted[columns].values       # shape (T_exp, 5)
    init_state = np.zeros((8,), dtype=np.float32)
    init_state[0] = y_seq[0, 0]
    initial_states_list.append(init_state)

print(initial_states_list[0])