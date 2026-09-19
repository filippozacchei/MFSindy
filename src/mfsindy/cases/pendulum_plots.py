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

def plot_multifidelity_trajectories(X_hf, X_lf, X_ref):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    for X in X_lf:
        ax.plot(
            X[::10, 0],
            X[::10, 1],
            ".",
            color="tab:red",
            alpha=0.1,
            linewidth=0.6,
        )

    for X in X_hf:
        ax.plot(
            X[::10, 0],
            X[::10, 1],
            ".",
            color="tab:blue",
            alpha=0.2,
            linewidth=0.8,
        )

    ax.plot(
        X_ref[0][:, 0],
        X_ref[0][:, 1],
        "-",
        color="black",
        alpha=1.0,
        linewidth=1.0,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)

    plt.tight_layout(pad=0.05)
    plt.show()


def plot_trajectories_additive_noise(x_true, alpha=0.15, sigma0=0.0):
    """Plot a clean pendulum trajectory together with state-dependent noisy samples."""
    x_true = np.asarray(x_true)

    omega_mag = np.abs(x_true[:, 1:2])
    std = sigma0 + alpha * omega_mag
    x_noisy = x_true + std * np.random.randn(*x_true.shape)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.plot(x_true[:, 0], x_true[:, 1], "-", color="black", linewidth=1.0)
    ax.scatter(
        x_noisy[:, 0],
        x_noisy[:, 1],
        s=4,
        color="tab:blue",
        alpha=0.35,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    return x_noisy


def plot_residuals(x_true, x_noisy):
    """Plot the absolute residual of the first pendulum component."""
    x_true = np.asarray(x_true)
    x_noisy = np.asarray(x_noisy)
    eps_x = np.abs(x_noisy[:, 0] - x_true[:, 0])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(eps_x, lw=0.9, color="#b32425")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.0, length=4)
    ax.grid(False)
    plt.tight_layout()
    plt.show()
