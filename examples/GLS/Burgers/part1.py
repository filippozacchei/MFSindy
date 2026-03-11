# %% part1.py
"""
Multi-fidelity SINDy on the 1D Burgers equation.

- generates low- and high-fidelity Burgers trajectories
- fits four ensemble SINDy / PDE-SINDy models:
    * HF only
    * LF only
    * MF   (HF + LF, unweighted)
    * MF_w (HF + LF, variance-weighted via pre-scaling)
- compares empirical posterior distributions of coefficients via
  Monte Carlo over random seeds:
    * MAE wrt true Burgers coefficients
    * mean L0 (support) mismatch
- visualizes the error distributions with 1D bubble histograms.
"""
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from utils import BurgersMFConfig, run_burgers_mf_experiment
from plot_utils import bubble_hist

warnings.filterwarnings("ignore")
sns.set(context="paper", style="white", font_scale=1.1)

# %%
cfg = BurgersMFConfig(results_filename="temp",n_runs=1)

print(
    f"Using {cfg.n_lf} LF trajectories and {cfg.n_hf} HF trajectories "
    f"(rel. noise LF={cfg.noise_lf_rel}, HF={cfg.noise_hf_rel})"
)
# %%
(
    df_errors,
    mae_errors,
    l0_errors,
    state_std,
    noise_hf_abs,
    noise_lf_abs,
) = run_burgers_mf_experiment(cfg)

print("Reference state std (scalar):", state_std)
print("HF noise (abs):", noise_hf_abs)
print("LF noise (abs):", noise_lf_abs)

# %% -------------------------------------------------------------------
# Compact Burgers panel for the figure table (HF / LF / reference slice)
# ---------------------------------------------------------------------
from utils import generate_burgers_dataset  # adjust import if needed
import matplotlib.pyplot as plt

seed_slice = 123

# regenerate HF / LF datasets using the same *absolute* noise levels
X_hf_noisy, t_train, x_grid, _ = generate_burgers_dataset(
    n_traj=cfg.n_hf,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=noise_hf_abs,
    seed=cfg.seed_base,
    L=cfg.L,
    NX=cfg.NX,
    nu=cfg.nu
    # add any extra Burgers-specific parameters here (e.g. nu, L)
)

X_lf_noisy, _, _, _ = generate_burgers_dataset(
    n_traj=cfg.n_lf,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=noise_lf_abs,
    seed=cfg.seed_base,
    L=cfg.L,
    NX=cfg.NX,
    nu=cfg.nu
)

# clean reference trajectory (no noise, long time if you like)
X_ref, t_ref, x_ref,_ = generate_burgers_dataset(
    n_traj=1,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=0.0,
    seed=cfg.seed_base,
    L=cfg.L,
    NX=cfg.NX,
    nu=cfg.nu
)

# %% ---------------------------------------------------------------------
# Choose a single trajectory and a representative time slice
# ---------------------------------------------------------------------
traj_idx = 0

u_hf = X_hf_noisy[traj_idx][:, :, 0]   # (Nx, Nt)
u_lf = X_lf_noisy[traj_idx][:, :, 0]   # (Nx, Nt)
u_ref = X_ref[0][:, :, 0]              # (Nx, Nt)

x = np.squeeze(x_grid)                 # (Nx,)
t = np.squeeze(t_train)                # (Nt,)

Xg, Tg = np.meshgrid(x, t, indexing="ij")  # (Nx, Nt)

fig = plt.figure(figsize=(5, 3.5), dpi=150)
ax = fig.add_subplot(111, projection="3d")

# LF surface (red, more transparent)
ax.plot_surface(
    Xg,
    Tg,
    u_lf,
    color="tab:red",
    alpha=0.25,
    linewidth=0,
    antialiased=True,
    shade=False,
)

# HF surface (blue, less transparent)
ax.plot_surface(
    Xg,
    Tg,
    u_hf,
    color="tab:blue",
    alpha=0.45,
    linewidth=0,
    antialiased=True,
    shade=False,
)

# Reference as a thin black “ridge” at final time
k_ref = -1
ax.plot(
    x,
    np.full_like(x, t[k_ref]),
    u_ref[:, k_ref],
    color="black",
    linewidth=1.0,
)

# Compact panel: no ticks, no axes, no titles
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_axis_off()
ax.grid(False)

plt.tight_layout(pad=0.05)
plt.show()
# %% ---------------------------------------------------------------
# Reload CSV and reconstruct error dictionaries for plotting
# ------------------------------------------------------------------
errors_path = os.path.join(cfg.results_dir, cfg.results_filename)
df_errors = pd.read_csv(errors_path)

models = ["HF", "LF", "MF", "MF_w"]

mae_errors = {
    m: df_errors[
        (df_errors["model"] == m) & (df_errors["metric"] == "MAE")
    ]["value"].to_numpy()
    for m in models
}
l0_errors = {
    m: df_errors[
        (df_errors["model"] == m) & (df_errors["metric"] == "L0")
    ]["value"].to_numpy()
    for m in models
}
    
models = ["HF", "LF", "MF", "MF_w"]

print("\nMAE errors (first few):")
for m in models:
    print(m, np.median(mae_errors[m]))

print("\nL0 support errors (first few):")
for m in models:
    print(m, np.median(l0_errors[m]))

# %% ----------------------------------------------------------------
# Bubble-hist plots for MAE and L0
# ------------------------------------------------------------------
method_colors = {
    "HF":   "tab:blue",
    "LF":   "tab:orange",
    "MF":   "tab:green",
    "MF_w": "tab:red",
}
models = ["MF_w", "MF", "LF", "HF", ]
labels = ["WMF", "MF", "LF", "HF", ]

bubble_hist(
    errors_dict=mae_errors,
    title=f"Burgers multi-fidelity: MAE on coefficients ({cfg.n_runs} runs)",
    xlabel="MAE",
    n_bins=20,
    models_order=models,
    colors=method_colors,
    labels=labels
)

bubble_hist(
    errors_dict=l0_errors,
    title=f"Burgers multi-fidelity: $L_0$ support error ({cfg.n_runs} runs)",
    xlabel=r"$L_0$ mismatch",
    n_bins=8,
    models_order=models,
    colors=method_colors,
    labels=labels
)

# %%
