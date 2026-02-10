import sys
import sys
import math
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_trained_k(path):
    d = yaml.safe_load(open(path, "r"))
    keys = [f"k{i}" for i in range(1, 8)]
    return np.array([d[k] for k in keys], dtype=np.float32)

project_root = Path(__file__).resolve().parents[1]
cfg_dir = project_root / "config" / "trained_params.yaml"

# Load identified constants
c_eaq_min_max = [3.7112687e-12,5.462755e-11]
print(5.462755e-11/3.7112687e-12)
k = c_eaq_min_max[0] *load_trained_k(cfg_dir)

#k = k/np.max(k)
#print(k)

# Number of states
n = 7
# Number of observations
m = 1

# Constrict A
A = np.zeros((n,n))
for i in range(n):
    if i == 0:
        A[i,i] = -k[i]
    else:
        A[i,i] = -k[i]
        A[i,i-1] = k[i-1]

C = 2*np.ones((m,n))

# ---- LTI observability matrix O = [C; CA; CA^2; ...; CA^(n-1)] ----
O = np.zeros((m * n, n))
A_pow = np.eye(n)

for i in range(n):
    # block row i: C * A^i
    O[i*m:(i+1)*m, :] = C @ A_pow
    # update A^i -> A^(i+1)
    A_pow = A_pow @ A

# Rank test (numerical)
rank_O = np.linalg.matrix_rank(O)
print("rank(O) =", rank_O, "out of n =", n)

# Optional: singular values + condition number (numerical observability)
# SVD of observability matrix
U, s, Vt = np.linalg.svd(O, full_matrices=False)

V = Vt.T  # columns of V are state-space directions
cond_O = s[0] / s[-1] if s[-1] > 0 else np.inf
print("singular values(O) =", s)
print("cond(O) =", cond_O)

for i in range(n):
    print(f"\nMode {i+1}")
    print(f"  singular value: {s[i]:.3e}")
    print(f"  state direction: {V[:, i]}")

# Observable iff rank(O) == n
is_observable = (rank_O == n)
print("observable =", is_observable)
    

