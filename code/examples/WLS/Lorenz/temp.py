import numpy as np
import pysindy as ps
import sys
sys.path.append("../../../")
from utils.training import run_ensemble_sindy
from generator import generate_lorenz_data

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
n_lf = 50
n_hf = 5
noise_lf = 0.25
noise_hf = 0.01
threshold = 0.5
dt = 1e-3
generator = generate_lorenz_data
system_name = "lorenz"

# ------------------------------------------------------------
# Library
# ------------------------------------------------------------
library = ps.PolynomialLibrary(degree=2)

# ------------------------------------------------------------
# Repeated runs
# ------------------------------------------------------------
n_reps = 100
n_ens  = 100

all_mf  = []
all_mfw = []

for rep in range(n_reps):
    print(f"Run {rep+1}/{n_reps}")

    X_hf, grid_hf, t_hf = generator(n_traj=n_hf, noise_level=noise_hf, seed=1000+rep, T=0.1)
    X_lf, _,      t_lf = generator(n_traj=n_lf, noise_level=noise_lf, seed=2000+rep, T=0.1)

    X_mix = X_hf + X_lf
    t_mix = t_hf + t_lf

    weights = [(1/noise_hf)**2]*len(X_hf) + [(1/noise_lf)**2]*len(X_lf)

    model_mf, opt_mf = run_ensemble_sindy(
        X_mix, t_mix, threshold=threshold, library=library, n_models=n_ens
    )
    model_mfw, opt_mfw = run_ensemble_sindy(
        X_mix, t_mix, threshold=threshold, library=library, weights=weights, n_models=n_ens
    )

    C_mf_rep  = np.array([coef.T for coef in opt_mf.coef_list])
    C_mfw_rep = np.array([coef.T for coef in opt_mfw.coef_list])

    all_mf.append(C_mf_rep)
    all_mfw.append(C_mfw_rep)

C_mf  = np.vstack(all_mf)   
C_mfw = np.vstack(all_mfw)

print("Coefficient array shapes:", C_mf.shape, C_mfw.shape)

# ------------------------------------------------------------
# Identify top-4 coefficients
# ------------------------------------------------------------
mean_mfw = np.mean(np.abs(C_mfw), axis=0)
flat = mean_mfw.flatten()
idx_flat = np.argsort(flat)[-4:]
idx_terms, idx_states = np.unravel_index(idx_flat, mean_mfw.shape)

feature_names = library.get_feature_names()
selected_names = [f"{feature_names[i]} (eq {s+1})" for i, s in zip(idx_terms, idx_states)]

# Extract distributions
C_mf_sel  = C_mf[:, idx_terms, idx_states]
C_mfw_sel = C_mfw[:, idx_terms, idx_states]

# Ground truth Lorenz coefficients
C_true = np.zeros((9, 3))
C_true[0, 0] = -10.0
C_true[1, 0] = 10.0
C_true[0, 1] = 28.0
C_true[5, 1] = -1.0
C_true[1, 1] = -1.0
C_true[4, 2] = 1.0
C_true[2, 2] = -8.0/3.0

true_vals = np.array([C_true[i, s] for i, s in zip(idx_terms, idx_states)])

# ------------------------------------------------------------
# SAVE FOR EXTERNAL PLOTTING
# ------------------------------------------------------------
np.save("C_mf_sel.npy",  C_mf_sel)
np.save("C_mfw_sel.npy", C_mfw_sel)
np.save("true_vals.npy", true_vals)
np.save("idx_terms.npy", idx_terms)
np.save("idx_states.npy", idx_states)
np.save("C_mf_samples.npy",  C_mf)
np.save("C_mfw_samples.npy", C_mfw)
with open("selected_names.txt", "w") as f:
    for name in selected_names:
        f.write(name + "\n")

print("✅ Export complete:")
print("   - C_mf_sel.npy")
print("   - C_mfw_sel.npy")
print("   - true_vals.npy")
print("   - idx_terms.npy / idx_states.npy")
print("   - selected_names.txt")
