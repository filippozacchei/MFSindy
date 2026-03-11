# %% lorenz_part1.py
"""
Multi-fidelity SINDy on the Lorenz system.

- generates low- and high-fidelity Lorenz trajectories
- fits four ensemble SINDy models:
    * HF only
    * LF only
    * MF   (HF + LF, unweighted)
    * MF_w (HF + LF, variance-weighted via pre-scaling)
- runs a Monte Carlo over random seeds to estimate:
    * MAE wrt true coefficients
    * mean L0 (support) mismatch
- saves the errors to disk and visualises them with bubble histograms.
"""

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from utils import LorenzMFConfig, run_lorenz_mf_experiment, generate_lorenz_dataset
from plot_utils import bubble_hist, animate_trajectories_rotating

warnings.filterwarnings("ignore")
sns.set(context="paper", style="white", font_scale=1.1)

# %% ---------------------------------------------------------------------
# Configuration and reference scaling
# ---------------------------------------------------------------------

cfg = LorenzMFConfig()

print(
    f"Using {cfg.n_lf} LF trajectories and {cfg.n_hf} HF trajectories "
    f"(rel. noise LF={cfg.noise_lf_rel}, HF={cfg.noise_hf_rel})"
)

# %% Optional: reference HF/LF datasets for visualisation / animation
X_hf_ref, t_train_ref, _ = generate_lorenz_dataset(
    n_traj=cfg.n_hf,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=cfg.noise_hf,
    seed=cfg.seed_base,
)
X_lf_ref, _, _ = generate_lorenz_dataset(
    n_traj=1,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=cfg.noise_lf,
    seed=cfg.seed_base,
)

print(f"HF (ref): {len(X_hf_ref)} trajectories, length {X_hf_ref[0].shape[0]}")
print(f"LF (ref): {len(X_lf_ref)} trajectories, length {X_lf_ref[0].shape[0]}")

# # %% Example: animate one true + HF/LF set (commented out by default)
# save_path = "videos/lorenz_hf_data.mp4"
# _ = animate_trajectories_rotating(
#     X_true_traj=X_lf_ref[0],
#     hf_traj=X_hf_ref[0],
#     lf_trajs=[],
#     n_frames=360,
#     elev=25,
#     azim_start=-60,
#     azim_step=1.0,
#     save_path=save_path,
# )

# %% ---------------------------------------------------------------------
# Run multi-fidelity Lorenz experiment (Monte Carlo + CSV export)
# ---------------------------------------------------------------------

(
    _df_errors_in_mem,
    _mae_errors_in_mem,
    _l0_errors_in_mem,
    state_std,
    noise_hf_abs,
    noise_lf_abs,
) = run_lorenz_mf_experiment(cfg)

print("Reference state std (scalar):", state_std)
print("HF noise (abs):", noise_hf_abs)
print("LF noise (abs):", noise_lf_abs)

# %% -------------------------------------------------------------------
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

print("\nMAE errors (first few, reloaded from CSV):")
for m in models:
    print(m, np.median(mae_errors[m]))

print("\nL0 support errors (first few, reloaded from CSV):")
for m in models:
    print(m, np.median(l0_errors[m]))

# %% -------------------------------------------------------------------
# Lorenz: 3D HF / LF trajectories with corresponding noise
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d proj)

seed = 100

# HF trajectories (noisy)
X_hf_noisy, t_train, _ = generate_lorenz_dataset(
    n_traj=10,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=noise_hf_abs,
    seed=cfg.seed_base,
)

# LF trajectories (noisy)
X_lf_noisy, _, _ = generate_lorenz_dataset(
    n_traj=100,
    T=cfg.T_train,
    dt=cfg.dt,
    noise_level=noise_lf_abs,
    seed=cfg.seed_base + 100,
)

# Long clean reference trajectory
X_ref, _, _ = generate_lorenz_dataset(
    n_traj=1,
    T=100,
    dt=cfg.dt,
    noise_level=0.0,
    seed=seed,
)

fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111, projection="3d")

# Clean reference (black curve)
ax.plot(
    X_ref[0][:, 0],
    X_ref[0][:, 1],
    X_ref[0][:, 2],
    "-",
    color="black",
    alpha=0.1,
    linewidth=1.0,
)


# LF trajectories (noisy, red)
for X in X_lf_noisy:
    ax.plot(
        X[:, 0],  # x
        X[:, 1],  # y
        X[:, 2],  # z
        ".",
        color="tab:red",
        alpha=0.2,
        linewidth=0.4,
        markersize=0.8,
    )

# HF trajectories (noisy, blue)
for X in X_hf_noisy:
    ax.plot(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        ".",
        color="tab:blue",
        alpha=0.6,
        linewidth=0.6,
        markersize=0.8,
    )


# Remove *everything* axis-related
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)

ax.xaxis.line.set_visible(False)
ax.yaxis.line.set_visible(False)
ax.zaxis.line.set_visible(False)

ax.grid(False)
ax.set_axis_off()   # hide the 3D box frame

plt.show()

# %% ---------------------------------------------------------------------
# Bubble-hist plots for MAE and L0
# ---------------------------------------------------------------------

models = ["MF_w", "MF", "LF", "HF", ]
labels = ["WMF", "MF", "LF", "HF", ]


method_colors = {
    "HF":   "tab:blue",
    "LF":   "tab:orange",
    "MF":   "tab:green",
    "MF_w":  "tab:red",
}

bubble_hist(
    errors_dict=mae_errors,
    title=f"Hopf multi-fidelity: MAE on coefficients ({cfg.n_runs} runs)",
    xlabel="MAE",
    n_bins=20,
    models_order=models,
    colors=method_colors,
    labels=labels
)

bubble_hist(
    errors_dict=l0_errors,
    title=f"Hopf multi-fidelity: $L_0$ support error ({cfg.n_runs} runs)",
    xlabel=r"$L_0$ mismatch",
    n_bins=8,
    models_order=models,
    colors=method_colors,
    labels=labels
    
)

# %%
