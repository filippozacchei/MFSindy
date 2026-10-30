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


def plot_initial_conditions(grid, flows, row_labels=None, save_path=None):
    """Initial u, v and rho of several flows, one row per flow.

    The 2D counterpart of the black initial-condition curves in the Burgers
    figure: what differs between the flows of a run is the initial condition
    (amplitudes and wavenumbers), so this is what the panel shows. u and v share
    a diverging scale centred at zero, rho a sequential one, each column on one
    scale so the flows can be compared.
    """
    import os

    G = np.asarray(grid)
    ics = [np.asarray(f)[:, :, 0, :] if np.ndim(f) == 4 else np.asarray(f) for f in flows]
    extent = [G[..., 0].min(), G[..., 0].max(), G[..., 1].min(), G[..., 1].max()]
    if row_labels is None:
        row_labels = [f"flow {j + 1}" for j in range(len(ics))]

    vel = max(np.abs(ic[..., :2]).max() for ic in ics)
    rho_lo = min(ic[..., 2].min() for ic in ics)
    rho_hi = max(ic[..., 2].max() for ic in ics)
    columns = [
        (r"$u_0$", "RdBu_r", -vel, vel),
        (r"$v_0$", "RdBu_r", -vel, vel),
        (r"$\rho_0$", "Purples", rho_lo, rho_hi),
    ]

    n = len(ics)
    fig, axes = plt.subplots(n, 3, figsize=(4.2, 1.35 * n + 0.6), dpi=150, squeeze=False)
    images = [None] * 3
    for i, ic in enumerate(ics):
        for c, (title, cmap, lo, hi) in enumerate(columns):
            ax = axes[i, c]
            images[c] = ax.imshow(ic[..., c].T, origin="lower", extent=extent,
                                  cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(row_labels[i], fontsize=8)
    for c in range(3):
        cbar = fig.colorbar(images[c], ax=axes[:, c], orientation="horizontal",
                            fraction=0.04, pad=0.03, aspect=12)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_visible(False)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=True)
    plt.show()


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
