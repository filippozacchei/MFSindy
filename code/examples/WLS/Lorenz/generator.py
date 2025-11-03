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


# if __name__ == "__main__":

#     # -----------------------------------------------------------
#     # Background attractor (long, noise-free)
#     # -----------------------------------------------------------
#     t_ref, X_ref, _ = generate_lorenz_trajectory(
#         y0=np.array([1.0, 1.0, 20.0]),
#         T=40.0, dt=1e-3, noise_level=0.0, seed=0
#     )

#     # -----------------------------------------------------------
#     # Two distinct initial conditions for HF and LF
#     # -----------------------------------------------------------
#     y0_hf = np.array([-4.0, -4.0, 25.0])
#     y0_lf = np.array([5.0, 5.0, 22.0])

#     # High fidelity: fine resolution, low noise
#     t_hf, X_hf, _ = generate_lorenz_trajectory(
#         y0=y0_hf, T=1.0, dt=2e-3, noise_level=0.01, seed=10
#     )

#     # Low fidelity: coarse resolution, higher noise
#     t_lf, X_lf, _ = generate_lorenz_trajectory(
#         y0=y0_lf, T=1.0, dt=1e-2, noise_level=0.08, seed=10
#     )

#     # Highlight last part of each trajectory for comparison
#     n_seg = 200
#     X_hf_seg = X_hf[-n_seg:, :]
#     X_lf_seg = X_lf[-n_seg:, :]

#     # -----------------------------------------------------------
#     # Aesthetic settings
#     # -----------------------------------------------------------
#     col_hf = "#2d4f8b"   # HF: deep cool blue
#     col_lf = "#d95f02"   # LF: warm orange
#     col_bg = "gray" # background attractor

#     plt.rcParams.update({
#         "axes.facecolor": "white",
#         "savefig.transparent": True,
#         "lines.solid_capstyle": "round",
#         "lines.solid_joinstyle": "round",
#     })

#     # *** Smaller figure: suitable for 1-column layouts ***
#     fig, ax = plt.subplots(figsize=(4.0, 3.3), dpi=350)

#     # Background attractor
#     ax.plot(X_ref[:,0], X_ref[:,1],
#             color=col_bg, lw=0.5, alpha=0.55, zorder=1)

#     # High fidelity (smooth curve)
#     ax.plot(X_hf_seg[:,0], X_hf_seg[:,1],
#             color=col_hf, lw=2.0, alpha=0.9, zorder=3)

#     # Low fidelity (dots over smooth, contrast highlight)
#     ax.plot(X_lf_seg[:,0], X_lf_seg[:,1],
#             color=col_lf, lw=2.3, alpha=0.35, zorder=2)
#     ax.plot(X_lf_seg[:,0], X_lf_seg[:,1],
#             ".", color=col_lf, ms=4.5, alpha=0.85, zorder=4)

#     # Remove axes and borders
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     plt.tight_layout()
#     plt.savefig("lorenz_multifidelity_clean_small.svg",
#                 format="svg", dpi=350, bbox_inches="tight", transparent=True)
#     plt.show()
import matplotlib.pyplot as plt
import numpy as np

# New color palette — consistent across figures
colors = ["#0C3B5D", "#3A7CA5", "#F28F3B"]   # HF → MF → LF

# Noise levels corresponding to the three fidelities
lorenz_noise_levels = [1, 25, 50]  # (HF lowest noise → LF highest noise)

# --- Long clean attractor (background) ---
t_ref, X_ref, _ = generate_lorenz_trajectory(
    y0=np.array([1.0, 1.0, 20.0]),
    T=40.0, dt=1e-3,
    noise_level=0.0, seed=42
)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(X_ref[:,0], X_ref[:,1],
        color="lightgray", lw=0.7, alpha=0.8, zorder=1)

# --- Multi-fidelity short segments ---
for nl, c in zip(lorenz_noise_levels, colors):

    t_noisy, X_noisy, _ = generate_lorenz_trajectory(
        T=100, dt=1e-3, noise_level=nl*0.01, seed=nl*153330
    )

    t_clean, X_clean, _ = generate_lorenz_trajectory(
        T=100, dt=1e-3, noise_level=0.0, seed=nl*153330
    )

    ax.plot(X_clean[-300:,0], X_clean[-300:,1],
            color=c, lw=4, alpha=0.6, zorder=2)

    ax.plot(X_noisy[-300:,0], X_noisy[-300:,1],
            ".", ms=5, alpha=0.9, color=c, zorder=3)

ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("lorenz_multifidelity_projection_clean.svg",
            format="svg", dpi=300, bbox_inches="tight", transparent=True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

lorenz_noise_levels = [1, 25, 50]
colors = ["#0C3B5D", "#3A7CA5", "#F28F3B"]

fig, ax = plt.subplots(figsize=(8,5))

for nl, c in zip(lorenz_noise_levels, colors):

    t, X_noisy, _ = generate_lorenz_trajectory(
        T=0.3, dt=1e-3, noise_level=nl*0.01, seed=nl-1
    )

    _, X_clean, _ = generate_lorenz_trajectory(
        T=0.3, dt=1e-3, noise_level=0.0, seed=nl-1
    )

    ax.plot(t, X_clean[:,0], lw=1.5, ls="--", color=c, alpha=0.6, zorder=1)
    ax.plot(t, X_noisy[:,0], lw=2.0,  color=c, alpha=0.9, zorder=2,
            label=f"Noise level = {nl}%")

ax.set_xlabel("Time", fontsize=20, weight="bold")
ax.set_ylabel(r"$x(t)$", fontsize=20, weight="bold", rotation=-90, labelpad=30)
ax.yaxis.set_label_position("right")

ax.legend(frameon=False, fontsize=16, loc='upper left')
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("lorenz_x_time_clean.svg",
            format="svg", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
