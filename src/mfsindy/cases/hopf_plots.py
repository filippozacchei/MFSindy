# plot_utils.py
"""
Plotting utilities for Burgers and Lorenz experiments.
"""

import numpy as np
import matplotlib.pyplot as plt

COLORS_MODELS = {
    "HF":   "tab:blue",
    "LF":   "tab:orange",
    "MF":   "tab:green",
    "VHF": "tab:cyan",
    "VLF": "tab:olive",
    "PMF": "tab:purple",
    "VMF": "tab:brown",
    "MF_w": "tab:red",
    "HF_2": "black",
    "LF_2": "tab:blue",
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


def plot_multifidelity_trajectories(X_hf, X_lf, X_clean):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    # LF trajectories (noisy) – red
    for X in X_lf:
        ax.plot(
            X[::10, 0],
            X[::10, 1],
            ".",
            color="tab:red",
            alpha=0.1,
            linewidth=0.6,
        )

    # HF trajectories (noisy) – blue
    for X in X_hf:
        ax.plot(
            X[::10, 0],
            X[::10, 1],
            ".",
            color="tab:blue",
            alpha=0.2,
            linewidth=0.8,
        )

    # Clean reference trajectory – black
    ax.plot(
        X_clean[0][:, 0],
        X_clean[0][:, 1],
        "-",
        color="black",
        alpha=1.0,
        linewidth=1.0,
    )

    # Remove ticks and spines for a clean, compact panel
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)

    plt.tight_layout(pad=0.05)
    plt.show()



def plot_trajectories_additive_noise(x_true, alpha=0.05, mu = 1.0):
    
    sigma0 = 1e-2
    alpha = 0.15

    state_norm = np.linalg.norm(x_true, axis=1, keepdims=True)   # shape (T, 1)
    noise = (alpha * np.abs(state_norm-mu) + sigma0) * np.random.randn(*x_true.shape)

    x_noisy = x_true + noise

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    ax.plot(x_true[:, 0], x_true[:, 1], "-",
        color=COLORS_MODELS["HF_2"],)

    ax.scatter(x_noisy[:, 0], x_noisy[:, 1],
            s=3, color=COLORS_MODELS["LF_2"], 
            alpha=0.4, label="noisy")


    # Limit cycle r = 1
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta),
            "--", color="gray", linewidth=1.0, alpha=0.8, label="limit cycle")

    ax.set_aspect("equal", "box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_frame_on(False)

    plt.tight_layout()
    plt.show()
    return x_noisy

def plot_residuals(x_true, x_noisy):
    x_clean = x_true[:, 0]
    x_noisy = x_noisy[:, 0]

    eps_x = np.abs(x_noisy - x_clean)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 24,
    })

    steelred = "#b32425"   # same color style as theta residual plot

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    ax.plot(
        eps_x,
        lw=0.9,
        color=steelred,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.0, length=4)
    ax.grid(False)

    plt.tight_layout()
    plt.show()
