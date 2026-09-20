# plot_utils.py
"""
Plotting utilities for Burgers and Lorenz experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Consistent colour scheme for Lorenz MF plots
COLORS_MODELS = {
    "HF":   "tab:blue",
    "LF":   "tab:orange",
    "MF":   "tab:green",
    "VHF": "tab:cyan",
    "VLF": "tab:olive",
    "PMF": "tab:purple",
    "VMF": "tab:brown",
    "MF_w": "tab:red",
}


def set_dark_theme(rc=None):
    """Apply a dark theme suitable for Lorenz 3D plots."""
    base = {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "font.size": 12,
    }
    if rc is not None:
        base.update(rc)
    plt.rcParams.update(base)

def plot_multifidelity_trajectories(X_hf, X_lf, X_ref, x, t):
    traj_idx = 0
    print(X_ref)
    u_hf = X_hf[traj_idx][:, :, 0]   # (Nx, Nt)
    u_lf = X_lf[traj_idx][:, :, 0]   # (Nx, Nt)
    u_ref = X_ref[:4]             # (Nx, Nt)

    x = np.squeeze(x)                 # (Nx,)
    t = np.squeeze(t)                # (Nt,)

    Xg, Tg = np.meshgrid(x, t, indexing="ij")  # (Nx, Nt)

    fig = plt.figure(figsize=(5, 3.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

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

    k_ref = 0
    for j in range(4):
        ttt = u_ref[j][:, :, 0] 
        ax.plot(
            x,
            np.full_like(x, t[k_ref]),
            ttt[:, k_ref],
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


def plot_trajectories_additive_noise(
    x_true,
    x_grid,
    alpha=0.25,
    sigma0=0.0,
    time_indices=None,
):
    """Plot clean/noisy Burgers trajectories together on one panel."""
    u_true = np.asarray(x_true)
    x = np.asarray(x_grid).squeeze()

    if u_true.ndim == 3:
        u_true = u_true[:, :, 0]
    if u_true.shape[0] < u_true.shape[1]:
        u_true = u_true.T

    dx = x[1] - x[0]
    grad = (np.roll(u_true, -1, axis=1) - np.roll(u_true, 1, axis=1)) / (2.0 * dx)
    std = sigma0 + alpha * np.abs(grad)
    u_noisy = u_true + std * np.random.randn(*u_true.shape)

    t = np.arange(u_true.shape[0], dtype=float)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")

    fig = plt.figure(figsize=(5, 3.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        Xg,
        Tg,
        u_noisy.T,
        color="tab:red",
        alpha=0.25,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.plot_surface(
        Xg,
        Tg,
        u_true.T,
        color="tab:blue",
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    return u_noisy


def plot_residuals(x_true, x_noisy, x_grid, time_indices=None):
    """Plot Burgers residual magnitude as a 3D field over space and time."""
    u_true = np.asarray(x_true)
    u_noisy = np.asarray(x_noisy)
    x = np.asarray(x_grid).squeeze()
    if u_true.ndim == 3:
        u_true = u_true[:, :, 0]
    if u_noisy.ndim == 3:
        u_noisy = u_noisy[:, :, 0]
    if u_true.shape[0] < u_true.shape[1]:
        u_true = u_true.T
    if u_noisy.shape[0] < u_noisy.shape[1]:
        u_noisy = u_noisy.T

    eps = np.abs(u_noisy - u_true)
    t = np.arange(eps.shape[0], dtype=float)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")

    fig = plt.figure(figsize=(5, 3.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        Xg,
        Tg,
        eps.T,
        color="#b32425",
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.grid(False)
    plt.tight_layout(pad=0.05)
    plt.show()
