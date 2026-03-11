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
    "HF":   "#F72585",  # magenta
    "LF":   "#4CC9F0",  # cyan
    "MF":   "#FFCA3A",  # yellow
    "MF_w": "#80FF72",  # green
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


def animate_trajectories_rotating(
    X_true_traj,
    hf_traj,          # single HF trajectory: (N, 3)
    lf_trajs,        # list of LF trajectories
    n_frames=360,
    elev=25,
    azim_start=-60,
    azim_step=1.0,
    save_path=None,
    dpi=200,
):
    """
    Rotating 3D animation for Lorenz trajectories.

    - True trajectory: fully drawn (white line).
    - LF trajectories: fully drawn (cloud, e.g. dots).
    - HF trajectory: drawn progressively over frames as a line.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D

    set_dark_theme()

    fig = plt.figure(figsize=(9.6, 5.4), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # True trajectory (full)
    line_true, = ax.plot(
        X_true_traj[:, 0],
        X_true_traj[:, 1],
        X_true_traj[:, 2],
        lw=0.25,
        alpha=0.25,
        color="white",
        label="True",
    )

    # HF trajectory: will be updated progressively
    hf_line, = ax.plot(
        [], [], [],
        '.',
        lw=1.4,
        alpha=0.9,
        color=COLORS_MODELS.get("HF", "tab:red"),
        label="HF",
    )

    # LF trajectories (static)
    lf_lines = []
    for X in lf_trajs:
        l, = ax.plot(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            ".",
            markersize=1.5,
            alpha=0.4,
            color=COLORS_MODELS.get("LF", "tab:blue"),
        )
        lf_lines.append(l)

    # Axis limits with approximate equal aspect
    all_pts = np.vstack([X_true_traj, hf_traj] + lf_trajs)
    xyz_min = all_pts.min(axis=0)
    xyz_max = all_pts.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = 0.5 * np.max(xyz_max - xyz_min)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Strip panes / grid for cleaner look
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)

    # Legend
    legend_handles = [
        Line2D([0], [0], color="white", lw=2, label="True"),
        Line2D([0], [0], color=COLORS_MODELS.get("HF", "tab:red"),
               lw=1.5, label="HF"),
        Line2D(
            [0],
            [0],
            color=COLORS_MODELS.get("LF", "tab:blue"),
            marker=".",
            linestyle="None",
            markersize=6,
            label="LF",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        labelcolor="white",
    )

    N = hf_traj.shape[0]

    def init():
        ax.view_init(elev=elev, azim=azim_start)
        # start HF line empty
        hf_line.set_data([], [])
        hf_line.set_3d_properties([])
        return (line_true, hf_line, *lf_lines)

    def update(frame):
        # how many points of the HF trajectory to show
        k = max(1, int((frame + 1) / n_frames * N))
        hf_line.set_data(hf_traj[(k-1000):k, 0], hf_traj[(k-1000):k, 1])
        hf_line.set_3d_properties(hf_traj[(k-1000):k, 2])

        azim = azim_start + frame * azim_step
        ax.view_init(elev=elev, azim=azim)
        return (line_true, hf_line, *lf_lines)

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=40,
        blit=False,
    )

    if save_path is not None:
        anim.save(save_path, writer="ffmpeg", dpi=dpi)

    plt.show()
    return anim


