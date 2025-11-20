"""
lorenz_generator.py

Dynamic generator for Lorenz system trajectories.
Supports low-, high-, or multi-fidelity data generation on the fly.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Lorenz equations
# ---------------------------------------------------------------------
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Standard Lorenz system ODEs."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# ---------------------------------------------------------------------
# Single trajectory generator
# ---------------------------------------------------------------------
def generate_lorenz_trajectory(
    y0=None,
    T=10.0,
    dt=1e-3,
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0,
    noise_level=0.0,
    seed=None,
):
    """
    Generate one Lorenz trajectory (possibly noisy).

    Returns
    -------
    t : ndarray (Nt,)
        Time vector.
    X : ndarray (Nt, 3)
        Trajectory (x, y, z) with optional noise.
    Xdot : ndarray (Nt, 3)
        Time derivatives.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, dt)

    if y0 is None:
        y0 = rng.uniform([-10, -10, 20], [10, 10, 30])

    sol = solve_ivp(lorenz, (t[0], t[-1]), y0, t_eval=t,
                    args=(sigma, rho, beta),
                    method="LSODA", rtol=1e-10, atol=1e-12)
    
    X = sol.y.T
    Xdot = np.array([lorenz(ti, xi, sigma, rho, beta) for ti, xi in zip(sol.t, X)])
    
    if noise_level > 0:
        X += rng.normal(0, noise_level, size=X.shape)

    return t, X, Xdot


# ---------------------------------------------------------------------
# Multi-trajectory generator (for LF/HF/MF)
# ---------------------------------------------------------------------
def generate_lorenz_data(
    n_traj=1,
    T=10.0,
    dt=1e-3,
    noise_level=0.0,
    seed=42,
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0,
):
    """
    Generate multiple Lorenz trajectories, analogous to generate_compressible_flow.
    """
    rng = np.random.default_rng(seed)
    trajectories, derivatives, times = [], [], []

    for i in range(n_traj):
        y0 = rng.uniform([-10, -10, 20], [10, 10, 30])
        t, X, Xdot = generate_lorenz_trajectory(
            y0=y0, T=T, dt=dt,
            sigma=sigma, rho=rho, beta=beta,
            noise_level=noise_level, seed=seed+i
        )
        trajectories.append(X)
        derivatives.append(Xdot)
        times.append(t)

    return trajectories, times[0], times


# ---------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------
def plot_lorenz_3d(X, title="Lorenz Attractor", ax=None):
    """3D trajectory plot."""
    from mpl_toolkits.mplot3d import Axes3D
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    plt.show()


def animate_lorenz(X, t, save_path="lorenz.gif"):
    """Animate Lorenz trajectory in 3D."""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], lw=1.5)

    ax.set_xlim(np.min(X[:, 0]), np.max(X[:, 0]))
    ax.set_ylim(np.min(X[:, 1]), np.max(X[:, 1]))
    ax.set_zlim(np.min(X[:, 2]), np.max(X[:, 2]))
    ax.set_title("Lorenz trajectory")

    def update(frame):
        line.set_data(X[:frame, 0], X[:frame, 1])
        line.set_3d_properties(X[:frame, 2])
        return line,

    anim = FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)
    anim.save(save_path, writer="pillow", fps=30)
    plt.close(fig)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

"""
Multi-Fidelity Lorenz Trajectory Visualization (3D)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # ---------------------------------------------------------------------
# # Lorenz system
# # ---------------------------------------------------------------------
# def lorenz(t, state, sigma=10, rho=28, beta=8/3):
#     x, y, z = state
#     return [
#         sigma * (y - x),
#         x * (rho - z) - y,
#         x * y - beta * z
#     ]


# # ---------------------------------------------------------------------
# # Single trajectory generator
# # ---------------------------------------------------------------------
# def generate_lorenz_trajectory(T, dt, noise_level=0.0, seed=None):
#     rng = np.random.default_rng(seed)
#     t = np.arange(0, T, dt)
#     y0 = rng.uniform([-10, -10, 20], [10, 10, 30])

#     sol = solve_ivp(lorenz, (0, T), y0, t_eval=t,
#                     method="LSODA", rtol=1e-10, atol=1e-12)
#     X = sol.y.T

#     if noise_level > 0:
#         X += rng.normal(0, noise_level, X.shape)

#     return X


# # ---------------------------------------------------------------------
# # Fidelity parameters
# # ---------------------------------------------------------------------
# T = 1

# # High fidelity:
# n_hf = 5
# dt_hf = 1e-3
# noise_hf = 0.0

# # Multi-fidelity continuum:
# n_levels = 5   # number of fidelity levels
# dt_lf = 1e-3
# noise_levels = np.linspace(0.0, 2.0, n_levels)   # increasing noise


# # ---------------------------------------------------------------------
# # Generate HF trajectories
# # ---------------------------------------------------------------------
# hf_trajs = [
#     generate_lorenz_trajectory(T=T*5, dt=dt_hf, noise_level=noise_hf, seed=200+k)
#     for k in range(n_hf)
# ]

# # ---------------------------------------------------------------------
# # Generate multiple noisy fidelity trajectories
# # ---------------------------------------------------------------------
# multi_fidelity_trajs = [
#     generate_lorenz_trajectory(T=T, dt=dt_lf, noise_level=nl, seed=200+i)
#     for i, nl in enumerate(noise_levels)
# ]

# plt.rcParams.update({
#     "figure.facecolor": "white",
#     "axes.facecolor": "white",
#     "font.size": 14,
# })

# # Colormap with stronger perceptual separation LF -> HF
# cmap = plt.cm.plasma
# color_values = np.linspace(0.0, 0.75, n_levels)

# fig = plt.figure(figsize=(6, 5), dpi=350)
# ax = fig.add_subplot(111, projection="3d")

# # ------------------------------------------------------------
# # 1) True HF reference underlay (light gray, smooth, subtle)
# # ------------------------------------------------------------
# for X in hf_trajs:
#     ax.plot(X[:,0], X[:,1], X[:,2],
#             lw=1.1, alpha=0.30, color="#A9A9A9", zorder=1)

# # ------------------------------------------------------------
# # 2) Multi-fidelity trajectories (dots, ordered LF → HF visually)
# # ------------------------------------------------------------
# # Reverse order so low-fidelity (noisier) appears below HF-like
# for X, c in zip(multi_fidelity_trajs[::-1], color_values):
#     ax.plot(X[:,0], X[:,1], X[:,2],
#             ".", markersize=1.8, alpha=0.75, color=cmap(c), zorder=3)

# # ------------------------------------------------------------
# # Axes formatting
# # ------------------------------------------------------------

# # ------------------------------------------------------------
# # Remove default 3D axes completely
# # ------------------------------------------------------------
# ax.set_axis_off()

# # Remove ticks
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# # ------------------------------------------------------------
# # Custom 3D Axis Triad (drawn manually)
# # ------------------------------------------------------------
# # Choose a center point slightly below the cloud (auto guess)
# all_pts = np.vstack(multi_fidelity_trajs + hf_trajs)
# # Original center (median of point cloud)
# center = np.median(all_pts, axis=0)

# # Shift triad down-left-backwards a bit
# shift = np.array([-0.5, -0.5, -0.5])   # adjust strength here
# center = center + shift * np.ptp(all_pts, axis=0)

# axis_len = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.15

# # Axis endpoints
# x_end = center + np.array([axis_len, 0, 0])
# y_end = center + np.array([0, axis_len, 0])
# z_end = center + np.array([0, 0, axis_len])

# # Draw lines (thin, behind dots)
# ax.plot([center[0], x_end[0]], [center[1], x_end[1]], [center[2], x_end[2]],
#         color="black", lw=0.9, alpha=0.7, zorder=0)
# ax.plot([center[0], y_end[0]], [center[1], y_end[1]], [center[2], y_end[2]],
#         color="black", lw=0.9, alpha=0.7, zorder=0)
# ax.plot([center[0], z_end[0]], [center[1], z_end[1]], [center[2], z_end[2]],
#         color="black", lw=0.9, alpha=0.7, zorder=0)

# # Arrowheads (tiny, minimal)
# arrow_size = axis_len * 0.04
# ax.quiver(*x_end, *(x_end-center), color="black", length=arrow_size, normalize=True)
# ax.quiver(*y_end, *(y_end-center), color="black", length=arrow_size, normalize=True)
# ax.quiver(*z_end, *(z_end-center), color="black", length=arrow_size, normalize=True)

# # Axis labels (small + offset)
# ax.text(*(x_end + 0.03*axis_len), 'x', fontsize=12)
# ax.text(*(y_end + 0.03*axis_len), 'y', fontsize=12)
# ax.text(*(z_end + 0.03*axis_len), 'z', fontsize=12)
# # Remove grid
# ax.grid(False)

# # Remove background pane surfaces (the "gray volume")
# ax.xaxis.pane.set_visible(False)
# ax.yaxis.pane.set_visible(False)
# ax.zaxis.pane.set_visible(False)


# # Do not adjust limits — preserved exactly
# ax.view_init(elev=22, azim=-60)

# plt.tight_layout()
# plt.savefig("lorenz_multifidelity_3d.png",
#             format="png", dpi=600, bbox_inches="tight")

# # ------------------------------------------------------------
# # SECOND FIGURE: x(t) comparison across fidelities (clean aesthetic)
# # ------------------------------------------------------------

# plt.rcParams.update({
#     "figure.facecolor": "white",
#     "axes.facecolor": "white",
#     "font.size": 13,
#     "lines.solid_capstyle": "round",
#     "lines.solid_joinstyle": "round",
# })

# # Time vectors
# t_hf = np.arange(0, T*5, dt_hf)
# t_lf = np.arange(0, T, dt_lf)

# fig2, ax2 = plt.subplots(figsize=(5.8, 3.2), dpi=400)

# # --- LF noisy trajectories (soft with colormap gradient) ---
# for X, c in zip(multi_fidelity_trajs[::-1], color_values):
#     ax2.plot(t_lf, X[:,0],
#              lw=1.0, alpha=0.65, color=cmap(c))

# # --- HF reference (subtle, thin, dashed) ---
# for X in hf_trajs:
#     n_match = min(len(t_lf), len(X[:,0]))
#     ax2.plot(t_lf[:n_match], X[:n_match,0],
#              linestyle="--", lw=0.5, alpha=0.95, color="black")

# # ------------------------------------------------------------
# # Aesthetic axes formatting
# # ------------------------------------------------------------
# ax2.set_xlabel(r"$t$", labelpad=4)
# ax2.set_ylabel(r"$x(t)$", labelpad=8, rotation=0)
# ax2.yaxis.set_label_position("right")  # Label on the right for balance

# # Remove default spines
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)

# # Draw custom x-axis line
# x_min, x_max = ax2.get_xlim()
# y_zero = ax2.get_ylim()[0] - 0.02*(ax2.get_ylim()[1]-ax2.get_ylim()[0])  # small shift down

# ax2.plot([x_min, x_max], [y_zero, y_zero], color="black", lw=1.0)

# # Draw arrowhead
# ax2.annotate("",
#     xy=(x_max, y_zero), xytext=(x_max - 0.04*(x_max-x_min), y_zero),
#     arrowprops=dict(arrowstyle="->", lw=1.0, color="black")
# )

# # Replace x-label (moved upward slightly)
# ax2.set_xlabel(r"$t$", labelpad=-2)

# # Place y-label on the right (as before)
# ax2.set_ylabel(r"$x(t)$", labelpad=8, rotation=0)
# ax2.yaxis.set_label_position("left")

# ax2.grid(False)
# ax2.set_xticks([])
# ax2.set_yticks([])

# # --- LF noisy trajectories (soft with colormap gradient) ---
# line_handles = []
# for j, (X, c) in enumerate(zip(multi_fidelity_trajs[::-1], color_values)):
#     (ln,) = ax2.plot(t_lf, X[:,0],
#                      lw=1.0, alpha=0.65, color=cmap(c))
#     line_handles.append(ln)

# import matplotlib as mpl

# # Discrete colors sampled from the same colormap
# # Reverse both color values and noise levels to match visual order
# noise_levels_disp = noise_levels
# color_values_disp = color_values

# # Build discrete colormap in *displayed* order
# colors_discrete = [cmap(c) for c in color_values_disp]
# cmap_discrete = mpl.colors.ListedColormap(colors_discrete)

# # Define boundaries for each region
# bounds = np.linspace(noise_levels_disp.min(), noise_levels_disp.max(), n_levels+1)
# norm = mpl.colors.BoundaryNorm(bounds, cmap_discrete.N)

# # Create the discrete colorbar
# sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_discrete)
# sm.set_array([])

# cbar = fig2.colorbar(sm, ax=ax2, fraction=0.045, pad=0.0,
#                      ticks=noise_levels_disp)

# # Tick labels (choose one style below)
# # Numerical labels:
# cbar.set_ticklabels([f"{lvl:.2f}" for lvl in noise_levels_disp])

# cbar.set_ticks([])
# cbar.ax.tick_params(length=0)

# plt.tight_layout()
# plt.savefig("lorenz_multifidelity_xt_clean.png",
#             format="png", dpi=600, bbox_inches="tight")
