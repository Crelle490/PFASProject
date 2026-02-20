import csv
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from so3_init_opt_robustness_new_model import build_model_with_k, set_cell_extra_params, get_cell_param_dict, _get_cell, make_constant_u_traj, prebuild_grid_tensors
from helper_functions import load_initials, load_trained_k, load_constants, load_cell_params

curent_dir = Path(__file__).resolve().parent
root_dir = curent_dir.parent

data = {"Time" : [],
        "Experiment 6": [],
        "Experiment 7": [],
        "Experiment 8": []}

data_key_map = {"Time (min)":"Time", "Exp 6 (F- mg/L)":"Experiment 6", "Exp 7 (F- mg/L)":"Experiment 7", "Exp 8 (F- mg/L)":"Experiment 8"}
keyes = ["Time","Experiment 6","Experiment 7","Experiment 8"]

with open(root_dir / 'data' /'validation_experiments.csv',newline='') as csvfile:
    content = csv.reader(csvfile,delimiter=',')
    frist_row = True
    for row in content:
        i = 0
        if not frist_row:
            for key in keyes:
                if row[i].isdigit():
                    data_val = int(row[i])
                elif len(row[i])==0:
                    data_val = None
                else: 
                    data_val = float(row[i])
                data[key].append(data_val)
                i += 1
        frist_row = False
 
data["Time (s)"] = []
keyes.append("Time (s)")
for time in data["Time"]:
    data[keyes[-1]].append(time*60)

cfg_dir = root_dir / "config"
trained_k_yaml = cfg_dir / "trained_params.yaml"

t_final = data["Time (s)"][-2]

dt = 5.0

t_sim = np.arange(0.0, t_final, dt, dtype=np.float32)

t_len = t_sim.size

c_so3 = [0.01,0.006]

k = load_trained_k(trained_k_yaml)

model, init, (pH, c_cl_init, c_so3_init, c_pfas_init) = build_model_with_k(cfg_dir=cfg_dir, k_vec=k,t_sim=t_sim,dt=dt)
model.trainable = False
cell_params = load_cell_params(cfg_dir / "trained_cell_params.yaml")
set_cell_extra_params(model, cell_params)
cell_nom = get_cell_param_dict(model)
cell = _get_cell(model)

# Warm up compile once (important)
u_warm = make_constant_u_traj(t_len, 0.0, c_cl_init, c_pfas_init)
_ = model([u_warm, init], training=False)


# Prebuild coarse tensors
u, x0, c_tf = prebuild_grid_tensors(
    t_len, c_so3, c_cl=c_cl_init, c_pfoa0=c_pfas_init, initial_states=init
)
y = model([u, x0], training=False)

# Recompute to % deflourination
init_pfoa_c = 10e-6 # mol/L
max_F_ion_c = 15*init_pfoa_c # mol/L
max_F_ion_mg_L_c = max_F_ion_c*19.0*100
print(max_F_ion_mg_L_c)

data["Exp 7 %"] = []
for data_point in data["Experiment 7"][:]:
    if not data_point == None:
        data["Exp 7 %"].append(data_point*100/max_F_ion_mg_L_c)

data["Exp 8 %"] = []
for data_point in data["Experiment 8"][:]:
    if not data_point == None:
        data["Exp 8 %"].append(data_point*100/max_F_ion_mg_L_c)

print(data)

plt.figure()
plt.scatter(data[keyes[-1]][:-2],data["Exp 7 %"][:-1],marker="o")
plt.plot(t_sim,y[0,:])
plt.legend(["Data","Simulated"])
plt.xlabel("Time [s]")
plt.ylabel("% Defluorination")
plt.title("Degredation curve @10mM SO3")
plt.savefig("exp_val_10mM.png", dpi=300, bbox_inches="tight")

plt.figure()
plt.scatter(data[keyes[-1]][:-1],data["Exp 8 %"][:],marker="o")
plt.plot(t_sim,y[1,:])
plt.legend("Data","Simulated")
plt.xlabel("Time [s]")
plt.ylabel("% Defluorination")
plt.title("Degredation curve @6mM SO3")
plt.savefig("exp_val_6mM.png", dpi=300, bbox_inches="tight")
plt.show()

