import sys
import sys
from pathlib import Path

import numpy as np
import yaml
from generation_of_eaq import GenerationOfElectrons,load_yaml_params,validate_keys,REQUIRED_PHYS_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.ode_runtime import build_model_from_config, load_constants

def load_trained_k(path):
    d = yaml.safe_load(open(path, "r"))
    keys = [f"k{i}" for i in range(1, 8)]
    return np.array([d[k] for k in keys], dtype=np.float32)

# -----------------------------
# Model construction
# -----------------------------
def build_A_from_k(k: np.ndarray) -> np.ndarray:
    """A(k) lower bidiagonal:
       A[i,i] = -k[i]
       A[i,i-1] =  k[i-1]  (i>0)
    """
    n = len(k)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -k[i]
        if i > 0:
            A[i, i - 1] = k[i - 1]
    return A

def build_G_from_x(x: np.ndarray) -> np.ndarray:
    """G(x) = [ (dA/dk1)x, ..., (dA/dkn)x ] for the above A structure.
       Each k_j appears in:
         A[j,j]   = -k_j  -> contributes -x[j] to row j
         A[j+1,j] = +k_j  -> contributes +x[j] to row j+1 (if j<n-1)
    """
    n = len(x)
    G = np.zeros((n, n))
    for j in range(n):
        G[j, j] += -x[j]
        if j < n - 1:
            G[j + 1, j] += +x[j]
    return G

def simulate_x_traj(x0: np.ndarray, k: np.ndarray, g_seq: np.ndarray) -> np.ndarray:
    """Simulate x_{k+1} = g_k * A(k) * x_k for k=0..N-1."""
    n = len(k)
    N = len(g_seq)
    A = build_A_from_k(k)

    X = np.zeros((N + 1, n))
    X[0] = x0
    for t in range(N):
        X[t + 1] = g_seq[t] * (A @ X[t])
    return X

def augmented_Fbar(xk: np.ndarray, k: np.ndarray, gk: float) -> np.ndarray:
    """Linearized augmented transition matrix for z=[x;k] at time k:
         x_{k+1} = gk A(k) x_k
         k_{k+1} = k_k
       => Fbar = [[gk A, gk G(xk)],
                  [  0 ,   I   ]]
    """
    n = len(k)
    A = build_A_from_k(k)
    G = build_G_from_x(xk)

    Fbar = np.zeros((2 * n, 2 * n))
    Fbar[:n, :n] = gk * A
    Fbar[:n, n:] = gk * G
    Fbar[n:, n:] = np.eye(n)
    return Fbar

# -----------------------------
# LTV augmented observability (finite horizon)
# -----------------------------
def build_aug_observability_matrix(X: np.ndarray, k: np.ndarray, g_seq: np.ndarray, C: np.ndarray, N_h: int):
    """
    Build LTV observability matrix for augmented linearization along trajectory:
      z_{t+1} = Fbar_t z_t,    y_t = Cbar z_t
    with Cbar = [C, 0].
    Returns O_aug of shape ((m*N_h) x (2n)).
    """
    n = len(k)
    m = C.shape[0]
    Cbar = np.hstack([C, np.zeros((m, n))])  # m x (2n)

    # Compute row blocks: Cbar * Phi(t+i, t), for i=0..N_h-1
    O = np.zeros((m * N_h, 2 * n))
    Phi = np.eye(2 * n)  # Phi(t,t) = I

    # i = 0 block: Cbar
    O[0:m, :] = Cbar @ Phi

    for i in range(1, N_h):
        # propagate Phi = Fbar_{t+i-1} * Phi
        # We'll assume starting time t=0 for simplicity.
        Fbar_prev = augmented_Fbar(X[i - 1], k, g_seq[i - 1])
        Phi = Fbar_prev @ Phi
        O[i * m:(i + 1) * m, :] = Cbar @ Phi

    return O, Cbar

def augmented_observability_gramian(X: np.ndarray, k: np.ndarray, g_seq: np.ndarray, C: np.ndarray, N_h: int):
    """
    W_o = sum_{i=0}^{N_h-1} Phi_i^T Cbar^T Cbar Phi_i, with Phi_0 = I, Phi_{i+1} = Fbar_i Phi_i.
    """
    n = len(k)
    m = C.shape[0]
    Cbar = np.hstack([C, np.zeros((m, n))])  # m x (2n)

    Phi = np.eye(2 * n)
    Wo = np.zeros((2 * n, 2 * n))

    # i=0
    Wo += Phi.T @ (Cbar.T @ Cbar) @ Phi

    for i in range(1, N_h):
        Fbar_prev = augmented_Fbar(X[i - 1], k, g_seq[i - 1])
        Phi = Fbar_prev @ Phi
        Wo += Phi.T @ (Cbar.T @ Cbar) @ Phi

    return Wo

# -----------------------------
# Verification routine
# -----------------------------
def verify_identifiability_augmented(x0, k, g_seq, C, N_h=None):
    n = len(k)
    if N_h is None:
        N_h = 2 * n  # a reasonable default horizon for augmented dimension 2n

    # simulate trajectory
    X = simulate_x_traj(x0=x0, k=k, g_seq=g_seq)

    # build augmented observability matrix
    O_aug, Cbar = build_aug_observability_matrix(X=X, k=k, g_seq=g_seq, C=C, N_h=N_h)

    # SVD/rank
    U, s, Vt = np.linalg.svd(O_aug, full_matrices=False)
    tol = max(O_aug.shape) * np.finfo(float).eps * s[0]
    rank = np.sum(s > tol)

    print("Augmented observability matrix shape:", O_aug.shape)
    print("svd singular values (O_aug):", s)
    print("tol:", tol)
    print(f"rank(O_aug) = {rank} out of {2*n}")

    # Gramian check (often nicer numerically)
    Wo = augmented_observability_gramian(X=X, k=k, g_seq=g_seq, C=C, N_h=N_h)
    eig = np.linalg.eigvalsh(Wo)
    tol_w = max(Wo.shape) * np.finfo(float).eps * np.max(eig)
    rank_w = np.sum(eig > tol_w)

    print("\nAugmented observability Gramian eigenvalues:", eig)
    print("tol_w:", tol_w)
    print(f"rank(Wo) = {rank_w} out of {2*n}")

    # Interpretation
    if rank < 2*n or rank_w < 2*n:
        print("\n=> Augmented system is NOT fully observable along this trajectory/horizon.")
        print("   This indicates NOT all parameters are locally identifiable from this output.")
    else:
        print("\n=> Augmented system appears observable along this trajectory/horizon.")
        print("   This supports local identifiability (at this operating point/trajectory).")

    return {
        "O_aug": O_aug,
        "svals": s,
        "rank_O": int(rank),
        "Wo": Wo,
        "eig_Wo": eig,
        "rank_Wo": int(rank_w),
        "Cbar": Cbar,
    }

# -----------------------------
# Example usage (EDIT THESE)
# -----------------------------
if __name__ == "__main__":
    # --- Replace with your actual k ---
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "config" / "trained_params.yaml"
    """
    t_final = 1200
    DT = 1.0

    # RNN step times: dt, 2dt, ... (we'll prepend t=0 later)
    t_sim = np.arange(0.0, t_final + DT, DT, dtype=np.float32)  # includes 0..t_final
    # build_model_from_config in your template expects t_sim length = horizon steps (often without 0)
    # We'll pass the "step times" excluding 0 to match the usual rollout convention:
    t_sim_steps = t_sim[1:]  # 1,2,...,t_final

    cfg_dir = PROJECT_ROOT / "config"
    initial_file = "initial_conditions_with_so3.yaml"
    trained_k_yaml = cfg_dir / "trained_params.yaml"

    model, dummy_input, initial_states = build_model_from_config(
        cfg_dir=cfg_dir,
        trained_k_yaml=trained_k_yaml,
        t_sim=t_sim_steps,
        dt=DT,
        initial_states=None,
        initial_file=initial_file
    )

    model.call([dummy_input,initial_states])
    """
    # Load identified constants
    k = load_trained_k(cfg_dir)
    #k = k/np.max(k)

    n = len(k)

    # --- Replace x0 with your experiment's initial conditions (7 states) ---
    x0 = np.zeros(n)
    x0[0] = 1.0  # e.g., initial PFOA only (placeholder)

    # --- Build g_seq = C_eaq(u_k) over horizon (must be known/assumed) ---
    # If u_k is constant and g is constant, g_seq = g0 * ones.
    cfg_dir = PROJECT_ROOT / "config"
    params_file = PROJECT_ROOT / "config" / "physichal_paramters.yaml"
    constants = load_constants(cfg_dir)

    phys = load_yaml_params(params_file)
    validate_keys(phys, REQUIRED_PHYS_KEYS, name="physichal_paramters.yaml")
    constants = {**constants, **phys}

    initial_states = np.zeros((1, 8), np.float32)
    initial_states[0,0] = 9.074690e-07
    gen = GenerationOfElectrons(constants, c_so3=0,pH=5.7, dummy_initial_state=initial_states)

    N_steps = 100
    c_so3 = np.linspace(0,0.01,100)

    c_eaq = []

    for concentractions in c_so3:
        c_eaq.append(gen._fun(concentractions))
    
    #g_seq = 3.7112687e-12 * np.ones(N_steps)  # placeholder
    g_seq = c_eaq

    # --- Fluoride-only output: y = C x, with C = 2*ones(1,n) in your earlier code ---
    C = 2.0 * np.ones((1, n))

    results = verify_identifiability_augmented(x0=x0, k=k, g_seq=g_seq, C=C, N_h=2*n)