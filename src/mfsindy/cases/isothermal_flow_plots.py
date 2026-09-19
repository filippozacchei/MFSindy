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

def plot_multifidelity_trajectories(grid, U_hf, U_lf, U_ref):

    k_mid = 0  # fixed time index

    u_hf  = U_hf[:, :, k_mid, 2]   # (Nx, Ny)
    u_lf  = U_lf[:, :, k_mid, 2]   # (Nx, Ny)
    u_ref = U_ref[:,     :, k_mid, 2]    # (Nx, Ny)

    x = grid[:, :, k_mid, 0]   # (Nx, Ny)
    y = grid[:, :, k_mid, 1]   # (Nx, Ny)

    fig = plt.figure(figsize=(5, 3.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        x, y, u_lf,
        color="tab:red",
        alpha=0.25,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.plot_surface(
        x, y, u_hf,
        color="tab:blue",
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.plot(
        x[:, 0],
        y[:, 0],
        u_ref[:, 0],
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


def plot_trajectories_additive_noise(
    grid,
    u_true,
    alpha=0.025,
    sigma0=1e-3,
    time_indices=None,
    component=2,
):
    """Plot clean/noisy isothermal-flow fields together at the selected time."""
    U = np.asarray(u_true)
    G = np.asarray(grid)

    if U.ndim != 4:
        raise ValueError("u_true must have shape (N, N, Nt, 3)")
    if G.ndim != 4:
        raise ValueError("grid must have shape (N, N, Nt, 3)")

    t = G[0, 0, :, 2]
    dt = t[1] - t[0]
    u_t = (np.roll(U[..., 0], -1, axis=2) - np.roll(U[..., 0], 1, axis=2)) / (2.0 * dt)
    v_t = (np.roll(U[..., 1], -1, axis=2) - np.roll(U[..., 1], 1, axis=2)) / (2.0 * dt)
    deriv_mag = np.sqrt(U[..., 0]**2 + U[..., 1]**2 + 0*U[..., 2]**2)
    std = sigma0 + alpha * deriv_mag
    U_noisy = U + std[..., None] * np.random.randn(*U.shape)

    n_t = U.shape[2]
    if time_indices is None:
        time_indices = [1]
    time_indices = [min(max(int(k), 0), n_t - 1) for k in time_indices[:1]]

    fig = plt.figure(figsize=(5, 3.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    k_idx = time_indices[0]
    x = G[:, :, k_idx, 0]
    y = G[:, :, k_idx, 1]
    clean = U[:, :, k_idx, component]
    noisy = U_noisy[:, :, k_idx, component]

    ax.plot_surface(
        x,
        y,
        noisy,
        color="tab:red",
        alpha=0.25,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.plot_surface(
        x,
        y,
        clean,
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
    plt.tight_layout(pad=0.05)
    plt.show()
    return U_noisy


def plot_residuals(grid, x_true, x_noisy, time_indices=None, component=2):
    """Plot isothermal-flow residual magnitude at the selected time instant."""
    G = np.asarray(grid)
    U = np.asarray(x_true)
    U_noisy = np.asarray(x_noisy)
    eps = np.abs(U_noisy[..., component] - U[..., component])
    n_t = eps.shape[2]
    if time_indices is None:
        time_indices = [1]
    k_idx = min(max(int(time_indices[0]), 0), n_t - 1)

    fig = plt.figure(figsize=(5, 3.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    X = G[:, :, k_idx, 0]
    Y = G[:, :, k_idx, 1]
    ax.plot_surface(
        X,
        Y,
        eps[:, :, k_idx],
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
    plt.tight_layout()
    plt.show()
