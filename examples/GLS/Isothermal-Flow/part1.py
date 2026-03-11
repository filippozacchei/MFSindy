# %% ns_part1.py
"""
Isothermal compressible Navier–Stokes — Part 1:
Multi-fidelity weak SINDy (HF / LF / MF / MF_w).

- Generates HF and LF noisy NS trajectories (homoscedastic; LF noisier)
- Fits four ensemble SINDy models:
    * HF   (HF only)
    * LF   (LF only)
    * MF   (HF + LF, unweighted)
    * MF_w (HF + LF, variance-weighted via pre-scaling)
- Runs a Monte Carlo over random seeds / ICs
- Computes:
    * MAE wrt reference NS coefficients (on true support)
    * mean L0 (support) mismatch
- Saves errors to CSV and visualises them via bubble histograms.
"""

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from ns_isothermal_utils import (
    NSIsothermalMFConfig,
    run_ns_isothermal_mf_experiment,
)
from plot_utils import bubble_hist

warnings.filterwarnings("ignore")
sns.set(context="paper", style="white", font_scale=1.1)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

cfg = NSIsothermalMFConfig(
    n_runs=1,          # increase if compute budget allows
    n_lf=1,
    n_hf=1,
    noise_lf_rel=0.025,
    noise_hf_rel=0.001,
    N=64,
    Nt=500,            # keep moderate for part1
    Nt_std=500,            # keep moderate for part1
    L=5.0,
    T=0.5,
    T_std = 0.5,
    mu=1.0,
    RT=1.0,
    derivative_order=2,
    include_bias=False,
    p=2,
    K=100,
    K_std=500,
    stlsq_threshold=0.5,
    stlsq_alpha=1e-8,
    n_ensemble_models=100,
    seed_base=2,
    results_dir="results",
    results_filename="temp.csv",
)

print(
    f"Using {cfg.n_lf} LF trajectories and {cfg.n_hf} HF trajectories "
    f"(rel. noise LF={cfg.noise_lf_rel}, HF={cfg.noise_hf_rel})"
)

# %% ---------------------------------------------------------------------
# Run multi-fidelity NS experiment
# ---------------------------------------------------------------------

(
    _df_errors_in_mem,
    _mae_errors_in_mem,
    _l0_errors_in_mem,
    state_std,
    noise_hf_abs,
    noise_lf_abs,
) = run_ns_isothermal_mf_experiment(cfg)

print("Reference state std (scalar):", state_std)
print("HF noise (abs):", noise_hf_abs)
print("LF noise (abs):", noise_lf_abs)

# ---------------------------------------------------------------------
# Reload CSV and reconstruct error dicts for plotting
# ---------------------------------------------------------------------

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

print("\nMedian MAE errors (reloaded from CSV):")
for m in models:
    print(m, np.median(mae_errors[m]))

print("\nMedian L0 support errors (reloaded from CSV):")
for m in models:
    print(m, np.median(l0_errors[m]))

# ---------------------------------------------------------------------
# Bubble-hist plots for MAE and L0
# ---------------------------------------------------------------------

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
    title=f"Isothermal NS multi-fidelity: MAE on coefficients ({cfg.n_runs} runs)",
    xlabel="MAE",
    n_bins=20,
    models_order=models,
    colors=method_colors,
    labels=labels,
)

bubble_hist(
    errors_dict=l0_errors,
    title=f"Isothermal NS multi-fidelity: $L_0$ support error ({cfg.n_runs} runs)",
    xlabel=r"$L_0$ mismatch",
    n_bins=10,
    models_order=models,
    colors=method_colors,
    labels=labels
)

# %% -------------------------------------------------------------------
# Compact 3D HF/LF panel for isothermal NS (centreline slice)
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from ns_isothermal_utils import generate_isothermal_ns_dataset

# --- Reference dataset just for noise scaling -------------------------
U_ref, t_ref, grid_ref = generate_isothermal_ns_dataset(
    N=cfg.N,
    Nt=cfg.Nt,
    L=cfg.L,
    T=cfg.T,
    mu=cfg.mu,
    RT=cfg.RT,
    seed=cfg.seed_base,
    ic_type="taylor-green",
)
state_std    = float(np.std(U_ref))
noise_hf_abs = cfg.noise_hf_rel * state_std
noise_lf_abs = cfg.noise_lf_rel * state_std

# --- One HF and one LF trajectory with homoscedastic noise -----------
rng_hf = np.random.default_rng(cfg.seed_base + 1000)
rng_lf = np.random.default_rng(cfg.seed_base + 2000)

U_hf_clean, t, grid = generate_isothermal_ns_dataset(
    N=cfg.N,
    Nt=cfg.Nt,
    L=cfg.L,
    T=cfg.T,
    mu=cfg.mu,
    RT=cfg.RT,
    seed=cfg.seed_base + 10,
    ic_type="taylor-green",
)
U_lf_clean, _, _ = generate_isothermal_ns_dataset(
    N=cfg.N,
    Nt=cfg.Nt,
    L=cfg.L,
    T=cfg.T,
    mu=cfg.mu,
    RT=cfg.RT,
    seed=cfg.seed_base + 20,
    ic_type="taylor-green",
)

U_hf_noisy = U_hf_clean + noise_hf_abs * rng_hf.standard_normal(U_hf_clean.shape)
U_lf_noisy = U_lf_clean + noise_lf_abs * rng_lf.standard_normal(U_lf_clean.shape)

# %% --- Fixed-time slice of u-velocity (compact 3D HF/LF panel) --------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# choose a representative time index (e.g. middle of the window)
k_mid = 0  # fixed time index

# u(x,y,t_k) for HF / LF / reference
u_hf  = U_hf_noisy[:, :, k_mid, 0]   # (Nx, Ny)
u_lf  = U_lf_noisy[:, :, k_mid, 0]   # (Nx, Ny)
u_ref = U_ref[:,     :, k_mid, 0]    # (Nx, Ny)

# corresponding spatial grid at that time
x = grid[:, :, k_mid, 0]   # (Nx, Ny)
y = grid[:, :, k_mid, 1]   # (Nx, Ny)

# --- 3D compact panel -------------------------------------------------
fig = plt.figure(figsize=(5, 3.5), dpi=150)
ax = fig.add_subplot(111, projection="3d")

# LF surface (red, more transparent)
ax.plot_surface(
    x, y, u_lf,
    color="tab:red",
    alpha=0.25,
    linewidth=0,
    antialiased=True,
    shade=False,
)

# HF surface (blue, less transparent)
ax.plot_surface(
    x, y, u_hf,
    color="tab:blue",
    alpha=0.45,
    linewidth=0,
    antialiased=True,
    shade=False,
)

# Reference ridge along a line (e.g. mid-y)
j_mid = cfg.N // 2
ax.plot(
    x[:, j_mid],
    y[:, j_mid],
    u_ref[:, j_mid],
    color="black",
    linewidth=1.0,
)

# Remove ticks / axes for table-friendly panel
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_axis_off()
ax.grid(False)

plt.tight_layout(pad=0.05)
plt.show()

# %%
