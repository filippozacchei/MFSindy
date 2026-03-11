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

def bubble_hist(
    errors_dict,
    title,          # kept for compatibility, but unused
    xlabel,         # kept for compatibility, but unused
    n_bins=8,
    models_order=None,
    colors=None,
    labels=None,
):
    """
    Compact 1D bubble-histogram plot for multiple methods.

    No titles or axis labels are added (for paper figure grids).
    """
    sns.set(style="white", context="paper")

    if models_order is None:
        models = list(errors_dict.keys())
    else:
        models = models_order

    # Colours
    if colors is None:
        try:
            palette = [COLORS_MODELS[m] for m in models]
            colors = {m: c for m, c in zip(models, palette)}
        except Exception:
            palette = sns.color_palette("tab10", n_colors=len(models))
            colors = {m: c for m, c in zip(models, palette)}

    all_vals = np.concatenate([np.asarray(errors_dict[m]) for m in models])
    vmin, vmax = all_vals.min(), all_vals.max()
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    bins = np.linspace(vmin - pad, vmax + pad, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # More compact figure
    fig, ax = plt.subplots(figsize=(3.0, 1.4), dpi=300)

    max_count = 0
    counts_per_model = {}
    for m in models:
        vals = np.asarray(errors_dict[m])
        counts, _ = np.histogram(vals, bins=bins)
        counts_per_model[m] = counts
        if counts.max() > max_count:
            max_count = counts.max()

    # Smaller bubbles for compact layout
    s_min = 10.0
    s_max = 120.0

    for i, m in enumerate(models):
        counts = counts_per_model[m]
        mask = counts > 0
        if not np.any(mask):
            continue

        freq = counts[mask].astype(float)
        sizes = s_min + (freq / max_count) * (s_max - s_min)
        x = centers[mask]
        y = np.full_like(x, i, dtype=float)

        ax.scatter(
            x,
            y,
            s=sizes,
            color=colors.get(m, "C0"),
            alpha=0.8,
            edgecolor="k",
            linewidth=0.3,
            zorder=3,
        )

    # Y: model labels (small font)
    if labels is None:
        labels = models
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=7)

    # X: keep ticks but small and tight
    ax.tick_params(axis="x", labelsize=7, pad=1)
    ax.tick_params(axis="y", pad=1)

    # Light vertical grid only
    ax.grid(axis="y", alpha=0.0)
    ax.grid(axis="x", alpha=0.2, linewidth=0.4)

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # No titles / axis labels here (for use in grids)
    # ax.set_title(...)
    # ax.set_xlabel(...)

    plt.tight_layout(pad=0.2)
    plt.show()
