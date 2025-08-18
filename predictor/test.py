# test.py - used to test kinetic model
# 

#### Importing necessary libraries ####
import os, sys
import numpy as np
from pathlib import Path
from ode_runtime import build_model_from_config, load_trained_k
from jacobian import build_jacobian_from_config
import yaml 
from EKF import ExtendedKalmanFilter
import matplotlib.pyplot as plt


here = Path(__file__).resolve().parent          # .../PFASProject/HPINN Predictor
project_root = here.parent                      # .../PFASProject

# so ode_runtime can find create_model:
model_dir = project_root / "Models Multiple Scripts" / "E_TF_MultipleBatch_Adaptive_c"
sys.path.insert(0, str(model_dir.resolve()))



#### Main Function ####
def main():
    # Random time serie
    t = np.arange(0, 5000, 1, dtype=np.float32)

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


    # ---- Build analytic Jacobian from config (minimal) ----

    k = load_trained_k(project_root / "config" / "trained_params.yaml")
    J = build_jacobian_from_config(project_root / "config", k)

    # Evaluate A (process Jacobian) and H (measurement Jacobian) at the initial state
    A0 = J.jacobian_reaction(x0[0])     # (8, 8)
    H0 = J.jacobian_observation(x0[0])  # (1, 8)

    print("A0 shape:", A0.shape, "H0 shape:", H0.shape)

    cov_path = project_root / "config" / "covariance_params.yaml"
    cov = yaml.safe_load(open(cov_path, "r"))

    Q  = np.array(cov["process_noise_covariance"],     dtype=np.float32)[:8, :8]
    R  = np.array(cov["measurement_noise_covariance"], dtype=np.float32)[:1, :1]
    P0 = np.array(cov["initial_error_covariance"],     dtype=np.float32)[:8, :8]

    print("Q shape:", Q.shape, "R shape:", R.shape, "P0 shape:", P0.shape)

    # measurement: z = [F-] = state[7]
    h = lambda x: np.array([x[7]], dtype=np.float32)
    H_jacobian = lambda x: J.jacobian_observation(x)      # (1,8)

    # process model (temporary stub = identity; swap in real one-step later)
    f = lambda x, u, k: x                                 # (8,)
    F_jacobian = lambda x: np.eye(8, dtype=np.float32)    # (8,8)

    # init EKF with initial state x0[0] and P0
    ekf = ExtendedKalmanFilter(
        f=f, h=h,
        F_jacobian=F_jacobian, H_jacobian=H_jacobian,
        Q=Q, R=R,
        x0=x0[0], P0=P0,
        k=k
    )
    print("EKF init → x:", ekf.x.shape, "P:", ekf.P.shape)

        # ---- EKF: run over a short synthetic F- stream (10 steps) ----
    rng = np.random.default_rng(0)
    z_stream = (y[0, 1:11, 7] + rng.normal(0.0, float(np.sqrt(R[0, 0])), size=10).astype(np.float32))  # (10,)
    for z in z_stream:
        ekf.predict(u=0.0)
        ekf.update(np.array([z], dtype=np.float32))
    print("EKF after 10 steps — F-:", float(ekf.x[7]), " C7:", float(ekf.x[0]))

    # ---- Plot simulated PFAS species and F- ----
   
    # y: (1, T, 8)
    Y = y[0]                              # (T, 8)
    T = Y.shape[0]
    # If you already have `t` from earlier, use it. Otherwise:
    t_plot = np.arange(T, dtype=np.float32)  # seconds, dt=1.0

    labels = ['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']

    plt.figure(figsize=(9, 6))
    # 5 PFAS panels
    for i, label in enumerate(labels):
        plt.subplot(3, 2, i + 1)
        plt.plot(t_plot, Y[:, i], label='Prediction')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.grid(True, alpha=0.3)
        plt.legend()

    # F- panel (bottom-right)
    plt.subplot(3, 2, 6)
    plt.plot(t_plot, Y[:, 7], label='F⁻')
    plt.xlabel('Time (s)')
    plt.ylabel('F⁻')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save figure
    results_dir = os.path.join(project_root, "results")  # or any folder you like
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "simulated_series.png")
    plt.savefig(out_path, dpi=5000, bbox_inches='tight')
    plt.close()

    print("Saved plot to:", out_path)


if __name__ == "__main__":
    main()




"""
# Usage example
k = [0.1, 0.2, 0.1, 0.05, 0.07, 0.03, 0.02]  # example rate constants
x0 = np.zeros(9)
P0 = np.eye(9) * 0.1
Q = np.eye(9) * 0.01
R = np.eye(1) * 0.1

jacobian = Jacobian(k)

ekf = ExtendedKalmanFilter(
    f=f,
    h=h,
    F_jacobian=jacobian.jacobian_reaction,
    H_jacobian=jacobian.jacobian_observation,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    k=k
)

u = 0.05  # some input affecting first state
z = np.array([0.2])  # measurement of fluoride concentration

ekf.predict(u)
ekf.update(z)

print("Updated state estimate:", ekf.x)


def build_jacobian_from_config(cfg_dir, k):
    cfg_dir = Path(cfg_dir)
    constants = yaml.safe_load(open(cfg_dir / "physichal_paramters.yaml", "r"))
    init      = yaml.safe_load(open(cfg_dir / "initial_conditions.yaml", "r"))

    pH   = float(init["pH"])
    c_cl = float(init["c_cl"])
    c_so3 = float(init["c_so3"])
    c_pfas_init = float(init["c_pfas_init"])

    # initial state: PFAS in slot 0, others 0 (shape (1,8))
    initial_state = np.zeros((1, 8), dtype=np.float32)
    initial_state[0, 0] = c_pfas_init

    return Jacobian(k, constants, pH, c_cl, c_so3, initial_state)
"""

"""
class Jacobian:
    def __init__(self, k):
        self.k = k
        self.x_dim = 9
        self.jacobian_reaction_calculate()

    def jacobian_reaction_calculate(self):
        x_syms = sp.symbols('x1:10')  # x1 to x9
        k_syms = sp.symbols('k1:8')   # k1 to k7

        r38 = k_syms[0] * x_syms[0] * x_syms[1]
        r39 = k_syms[1] * x_syms[0] * x_syms[2]
        r40 = k_syms[2] * x_syms[0] * x_syms[4]
        r41 = k_syms[3] * x_syms[0] * x_syms[5]
        r42 = k_syms[4] * x_syms[0] * x_syms[6]
        r43 = k_syms[5] * x_syms[0] * x_syms[7]
        r44 = k_syms[6] * x_syms[0] * x_syms[8]

        dx = [0]*9
        dx[0] = -r38 - r39 - r40 - r41 - r42 - r43 - r44
        dx[1] = -r38
        dx[2] = -r39
        dx[3] = 2 * (r38 + r39 + r40 + r41 + r42 + r43 + r44)
        dx[4] = r39 - r40
        dx[5] = r40 - r41
        dx[6] = r41 - r42
        dx[7] = r42 - r43
        dx[8] = r43 - r44

        J = sp.Matrix(dx).jacobian(x_syms)
        self._x_syms = x_syms
        self._k_syms = k_syms
        self.J_reaction_func = sp.lambdify((x_syms, k_syms), J, modules='numpy')

    def jacobian_reaction(self, x_point):
        return self.J_reaction_func(x_point, self.k)

    def jacobian_observation(self):
        H = np.zeros((1, self.x_dim))
        H[0, 3] = 1  # fluoride concentration at index 3
        return H
"""

"""
#class ExtendedKalmanFilter:
#    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0, k):
#        self.f = f
#        self.h = h
#        self.Q = Q
#        self.R = R
#        self.x = x0
#        self.P = P0
#        self.k = k
#        self.F_jacobian = F_jacobian
#        self.H_jacobian = H_jacobian
#
#    def predict(self, u):
##        F = self.F_jacobian(self.x)
 #       self.x = self.f(self.x, u, self.k)
#        self.P = F @ self.P @ F.T + self.Q
#
#    def update(self, z):
#        H = self.H_jacobian()
#        y = z - self.h(self.x)
#        S = H @ self.P @ H.T + self.R
#        K = self.P @ H.T @ np.linalg.inv(S)
 #       self.x = self.x + K @ y
#        I = np.eye(len(self.x))
#        self.P = (I - K @ H) @ self.P
"""

