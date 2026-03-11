# %% pendulum_part1.py
"""
Multi-fidelity SINDy on the damped single pendulum.

- generates low- and high-fidelity pendulum trajectories
- fits four ensemble SINDy models:
    * HF only
    * LF only
    * MF   (HF + LF, unweighted)
    * MF_w (HF + LF, variance-weighted via per-trajectory weights)
- runs a Monte Carlo over random seeds to estimate:
    * MAE wrt true pendulum coefficients (on true support)
    * mean L0 (support) mismatch
- saves the errors to disk and visualises them with bubble histograms.
"""

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pendulum_utils import (
    PendulumMFConfig,
    run_pendulum_mf_experiment,
    generate_pendulum_dataset,
)
from plot_utils import bubble_hist

warnings.filterwarnings("ignore")
sns.set(context="paper", style="white", font_scale=1.1)

# %%---------------------------------------------------------------------
# Configuration and reference datasets (optional visualisation)
# ---------------------------------------------------------------------

cfg = PendulumMFConfig(
    n_runs=1,
    T_train=0.1,
    T_true=10.0,
    n_lf=100,
    n_hf=10,
    g=9.81,
    L=1.0,
    c=0.5,
    stlsq_threshold=0.05,
    results_filename="temp"
)

print(
    f"Using {cfg.n_lf} LF trajectories and {cfg.n_hf} HF trajectories "
    f"(rel. noise LF={cfg.noise_lf_rel}, HF={cfg.noise_hf_rel})"
)

# Optional *noise-free* reference HF/LF datasets (for quick checks or plots)
X_hf_ref, t_train_ref, _ = generate_pendulum_dataset(
    n_traj=cfg.n_hf,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=0.0,
    seed=cfg.seed_base,
    g=cfg.g,
    L=cfg.L,
    c=cfg.c,
)
X_lf_ref, _, _ = generate_pendulum_dataset(
    n_traj=cfg.n_lf,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=0.0,
    seed=cfg.seed_base + 100,
    g=cfg.g,
    L=cfg.L,
    c=cfg.c,
)

print(f"HF (ref): {len(X_hf_ref)} trajectories, length {X_hf_ref[0].shape[0]}")
print(f"LF (ref): {len(X_lf_ref)} trajectories, length {X_lf_ref[0].shape[0]}")

# %% ---------------------------------------------------------------------
# Run pendulum multi-fidelity experiment (Monte Carlo + CSV export)
# ---------------------------------------------------------------------

(
    _df_errors_in_mem,
    _mae_errors_in_mem,
    _l0_errors_in_mem,
    state_std,
    noise_hf_abs,
    noise_lf_abs,
) = run_pendulum_mf_experiment(cfg)

print("Reference state std (scalar):", state_std)
print("HF noise (abs):", noise_hf_abs)
print("LF noise (abs):", noise_lf_abs)

# %% -------------------------------------------------------------------
# Phase-plane visualisation of HF / LF trajectories with corresponding noise
# ---------------------------------------------------------------------
seed = 100
# regenerate HF / LF datasets using the *absolute* noise levels used in MF experiment
X_hf_noisy, t_train, _ = generate_pendulum_dataset(
    n_traj=10,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=noise_hf_abs,
    seed=cfg.seed_base,
    g=cfg.g,
    L=cfg.L,
    c=cfg.c,
)
X_lf_noisy, _, _ = generate_pendulum_dataset(
    n_traj=100,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=noise_lf_abs,
    seed=cfg.seed_base + 100,
    g=cfg.g,
    L=cfg.L,
    c=cfg.c,
)
X_ref, _, _ = generate_pendulum_dataset(
    n_traj=1,
    T=100,
    dt=cfg.dt,
    noise_level=0.0,
    seed=seed,
    g=cfg.g,
    L=cfg.L,
    c=cfg.c,
)

fig, ax = plt.subplots(figsize=(5, 5), dpi=150)


# LF trajectories (noisy)
for X in X_lf_noisy:
    ax.plot(
        X[:, 0],  # theta
        X[:, 1],  # omega
        ".",
        color="tab:red",
        alpha=0.2,
        linewidth=0.6,
    )

# HF trajectories (noisy)
for X in X_hf_noisy:
    ax.plot(
        X[:, 0],
        X[:, 1],
        ".",
        color="tab:blue",
        alpha=0.6,
        linewidth=0.8,
    )

ax.plot(
    X_ref[0][:, 0],  # theta
    X_ref[0][:, 1],  # omega
    "-",
    color="black",
    alpha=1.0,
    linewidth=1,
)

# Remove ticks and spines for a clean look
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.grid(False)

plt.tight_layout(pad=0.05)
plt.show()
# %%
# # %% -------------------------------------------------------------------
# # Reload CSV and reconstruct error dicts for plotting
# # ---------------------------------------------------------------------

# errors_path = os.path.join(cfg.results_dir, cfg.results_filename)
# df_errors = pd.read_csv(errors_path)

# models = ["HF", "LF", "MF", "MF_w"]

# mae_errors = {
#     m: df_errors[
#         (df_errors["model"] == m) & (df_errors["metric"] == "MAE")
#     ]["value"].to_numpy()
#     for m in models
# }
# l0_errors = {
#     m: df_errors[
#         (df_errors["model"] == m) & (df_errors["metric"] == "L0")
#     ]["value"].to_numpy()
#     for m in models
# }

# print("\nMedian MAE errors (reloaded from CSV):")
# for m in models:
#     print(f"{m:4s} : {np.median(mae_errors[m]):.3e}")

# print("\nMedian L0 support errors (reloaded from CSV):")
# for m in models:
#     print(f"{m:4s} : {np.median(l0_errors[m]):.3e}")

# # %%---------------------------------------------------------------------
# # Bubble-hist plots for MAE and L0
# # ---------------------------------------------------------------------

# models = ["MF_w", "MF", "LF", "HF", ]
# labels = ["WMF", "MF", "LF", "HF", ]
# method_colors = {
#     "HF": "tab:blue",
#     "LF": "tab:orange",
#     "MF": "tab:green",
#     "MF_w": "tab:red",
# }

# bubble_hist(
#     errors_dict=mae_errors,
#     title=f"Pendulum multi-fidelity: MAE on coefficients ({cfg.n_runs} runs)",
#     xlabel="MAE",
#     n_bins=20,
#     models_order=models,
#     colors=method_colors,
#     labels=labels,
# )

# bubble_hist(
#     errors_dict=l0_errors,
#     title=f"Pendulum multi-fidelity: $L_0$ support error ({cfg.n_runs} runs)",
#     xlabel=r"$L_0$ mismatch",
#     n_bins=8,
#     models_order=models,
#     labels=labels,
#     colors=method_colors,
# )

# # %%

# %%
