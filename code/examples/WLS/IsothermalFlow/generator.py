"""
compressible_generator.py

Generator for the 2D isothermal compressible Navier–Stokes dataset.
Supports multiple independent trajectories for multi-fidelity learning.
"""

import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Compressible isothermal Navier–Stokes in 2D
# ---------------------------------------------------------------------
def compressible(t, U, dx, N, mu, RT):
    u = U.reshape(N, N, 3)[:, :, 0]
    v = U.reshape(N, N, 3)[:, :, 1]
    rho = U.reshape(N, N, 3)[:, :, 2]
    ux = ps.differentiation.FiniteDifference(
        d=1,
        axis=0,
        periodic=True,
    )._differentiate(u, dx)
    uy = ps.differentiation.FiniteDifference(
        d=1,
        axis=1,
        periodic=True,
    )._differentiate(u, dx)
    uxx = ps.differentiation.FiniteDifference(
        d=2,
        axis=0,
        periodic=True,
    )._differentiate(u, dx)
    uyy = ps.differentiation.FiniteDifference(
        d=2,
        axis=1,
        periodic=True,
    )._differentiate(u, dx)
    vx = ps.differentiation.FiniteDifference(
        d=1,
        axis=0,
        periodic=True,
    )._differentiate(v, dx)
    vy = ps.differentiation.FiniteDifference(
        d=1,
        axis=1,
        periodic=True,
    )._differentiate(v, dx)
    vxx = ps.differentiation.FiniteDifference(
        d=2,
        axis=0,
        periodic=True,
    )._differentiate(v, dx)
    vyy = ps.differentiation.FiniteDifference(
        d=2,
        axis=1,
        periodic=True,
    )._differentiate(v, dx)
    px = ps.differentiation.FiniteDifference(
        d=1,
        axis=0,
        periodic=True,
    )._differentiate(rho * RT, dx)
    py = ps.differentiation.FiniteDifference(
        d=1,
        axis=1,
        periodic=True,
    )._differentiate(rho * RT, dx)
    ret = np.zeros((N, N, 3))
    ret[:, :, 0] = -(u * ux + v * uy) - (px - mu * (uxx + uyy)) / rho
    ret[:, :, 1] = -(u * vx + v * vy) - (py - mu * (vxx + vyy)) / rho
    ret[:, :, 2] = -(u * px / RT + v * py / RT + rho * ux + rho * vy)
    return ret.reshape(3 * N * N)


# ---------------------------------------------------------------------
# Initial condition generator
# ---------------------------------------------------------------------
def make_initial_condition(X, Y, L, ic_type="taylor-green", perturb_scale=1):
    """Return (U0, V0, RHO0) for a chosen flow configuration."""
    rng = np.random.default_rng()
    if ic_type == "taylor-green":
        U0 = (-np.sin(2 * np.pi / L * X) + 0.5 * np.cos(2 * 2 * np.pi / L * Y))
        V0 = (0.5 * np.cos(2 * np.pi / L * X) - np.sin(2 * 2 * np.pi / L * Y))
        RHO0 = ( 1 + 0.5 * np.cos(2 * np.pi / L * X) * np.cos(2 * 2 * np.pi / L * Y))
        # U0 = np.sin(2 * np.pi * X / L) * np.cos(2 * np.pi * Y / L)
        # V0 = -np.cos(2 * np.pi * X / L) * np.sin(2 * np.pi * Y / L)
        # RHO0 = 1.0 + 0.1 * np.cos(4 * np.pi * X / L) * np.cos(4 * np.pi * Y / L)
        perturb = perturb_scale * np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (0.1 * L) ** 2) * (0.5*rng.standard_normal(U0.shape))
        U0 += perturb
        V0 -= perturb
        

    elif ic_type == "shear-layer":
        U0 = np.tanh((Y - L / 2) / 0.1)
        V0 = 0.05 * np.sin(2 * np.pi * X / L)
        RHO0 = 1.0 + 0.1 * np.exp(-((Y - L / 2) ** 2) / (0.1**2))

    else:
        raise ValueError(f"Unknown initial condition: {ic_type}")

    return U0, V0, RHO0


# ---------------------------------------------------------------------
# Flow field generator (multi-trajectory)
# ---------------------------------------------------------------------
# We need this signature
# X_hf, grid_hf, t_hf = generator(n_hf, 
#                                 noise_level=noise_level_hf * std_per_dim, 
#                                 T=T, 
#                                 seed=run*seed)
def generate_compressible_flow(
    n_traj=1,
    N=64,
    Nt=101,
    L=5,
    T=2,
    mu=1.0,
    RT=1.0,
    noise_level=0.0,
    seed=42,
    initial_condition="taylor-green",
    noise_0 = 0.1
):
    """
    Generate one or more trajectories for the 2D isothermal
    compressible Navier–Stokes equations.

    Parameters
    ----------
    n_traj : int
        Number of trajectories to simulate.
    N : int
        Grid points per spatial dimension.
    Nt : int
        Number of time snapshots per trajectory.
    L, T, mu, RT, noise_level : float
        Physical and numerical parameters (see single-trajectory doc).
    seed : int
        Random seed for reproducibility.
    initial_condition : str
        Either 'taylor-green' or 'shear-layer'.

    Returns
    -------
    trajectories : list of dict
        Each element is a dictionary containing:
        {
          "u_field": ndarray (N, N, Nt, 3),
          "u_dot": ndarray (N, N, Nt, 3),
          "grid": ndarray (N, N, Nt, 3),
          "t": ndarray (Nt,)
        }
    """

    rng = np.random.default_rng(seed)
    t = np.linspace(0, T, Nt)
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    trajectories = []
    ts = []

    for i in range(n_traj):
        # Slightly randomize the initial condition
        U0, V0, RHO0 = make_initial_condition(X, Y, L, ic_type=initial_condition)
        noise_ic = noise_0 * rng.standard_normal((N, N, 3))
        y0 = np.stack([U0, V0, RHO0], axis=-1) + noise_ic

        sol = solve_ivp(
            compressible,
            (t[0], t[-1]),
            y0=y0.reshape(-1),
            t_eval=t,
            args=(dx, N, mu, RT),
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )

        u_field = sol.y.reshape(N, N, 3, -1).transpose(0, 1, 3, 2)

        # Add measurement noise if desired
        if noise_level > 0.0:
            u_field += noise_level * rng.standard_normal(size=u_field.shape)

        # Construct spatiotemporal grid
        grid = np.zeros((N, N, Nt, 3))
        grid[:, :, :, 0] = x[:, None, None]
        grid[:, :, :, 1] = y[None, :, None]
        grid[:, :, :, 2] = t[None, None, :]

        trajectories.append(u_field)
        ts.append(t)

    return trajectories, grid, ts

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------
def plot_snapshot(u_field, t, L, idx=None, title_prefix=""):
    """
    Plot u, v, rho fields for a given snapshot in time.

    Parameters
    ----------
    u_field : ndarray
        Flow field of shape (N, N, Nt, 3)
    t : ndarray
        Time array
    L : float
        Domain length
    idx : int, optional
        Time index to visualize (default: middle frame)
    title_prefix : str
        Prefix for subplot titles
    """
    N = u_field.shape[0]
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if idx is None:
        idx = u_field.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fields = [u_field[:, :, idx, 0], u_field[:, :, idx, 1], u_field[:, :, idx, 2]]
    labels = [r"$u(x,y)$", r"$v(x,y)$", r"$\rho(x,y)$"]
    cmaps = ["RdBu_r", "RdBu_r", "viridis"]

    for ax, f, label, cmap in zip(axes, fields, labels, cmaps):
        im = ax.pcolormesh(X, Y, f, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{title_prefix}{label} @ t={t[idx]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()


def animate_field(u_field, t, L, var_index=0, title=None, save_path=None):
    """
    Animate a given component (u, v, or rho) in time.

    Parameters
    ----------
    u_field : ndarray
        Flow field of shape (N, N, Nt, 3)
    t : ndarray
        Time array
    L : float
        Domain length
    var_index : int
        0 for u, 1 for v, 2 for rho
    title : str
        Figure title
    save_path : str or None
        If provided, saves animation to given file path
    """
    N = u_field.shape[0]
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    var_name = ["u", "v", r"\rho"][var_index]
    field = u_field[:, :, :, var_index]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, Y, field[:, :, 0], cmap="RdBu_r", shading="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(var_name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title or 'Compressible Flow'} — {var_name} field")

    def update(frame):
        im.set_array(field[:, :, frame].ravel())
        ax.set_title(f"{title or 'Compressible Flow'} — {var_name} @ t={t[frame]:.3f}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(t), interval=80, blit=False)
    plt.tight_layout()

    if save_path:
        anim.save(save_path, dpi=200, fps=15)
        print(f"💾 Animation saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def compare_trajectories(trajectories, t, L, component=0, idx=None):
    """
    Compare multiple trajectories for the same variable at a fixed time index.

    Parameters
    ----------
    trajectories : list of ndarray
        Each element is u_field of shape (N, N, Nt, 3)
    t : ndarray
        Shared time array
    L : float
        Domain length
    component : int
        0 for u, 1 for v, 2 for rho
    idx : int or None
        Frame index to compare (default: mid-frame)
    """
    if idx is None:
        idx = trajectories[0].shape[2] // 2

    n_traj = len(trajectories)
    fig, axes = plt.subplots(1, n_traj, figsize=(5 * n_traj, 4))
    if n_traj == 1:
        axes = [axes]

    var_name = ["u", "v", r"\rho"][component]
    for i, (u_field, ax) in enumerate(zip(trajectories, axes)):
        im = ax.imshow(
            u_field[:, :, idx, component],
            origin="lower",
            cmap="RdBu_r",
            extent=(0, L, 0, L),
        )
        ax.set_title(f"Trajectory {i+1} — {var_name} @ t={t[idx]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_velocity_magnitude_surface_minimal(u_field, L, idx=0):
    """
    Minimal, clean 3D surface plot of the velocity magnitude |u|
    at a given time index (default: initial condition).
    """
    N = u_field.shape[0]
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Velocity magnitude
    u = u_field[:, :, idx, 0]
    v = u_field[:, :, idx, 1]
    vel_mag = np.sqrt(u**2 + v**2)

    fig = plt.figure(figsize=(5, 4), dpi=350)
    ax = fig.add_subplot(111, projection="3d")

    # Surface plot
    ax.plot_surface(
        X, Y, vel_mag,
        cmap=cm.viridis,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )

    # Remove grid, ticks, frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_box_aspect((1, 1, 0.4))  # flatter visual profile
    ax.grid(False)

    # Remove pane colors for clean white background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")

    plt.tight_layout()
    plt.savefig("compressible_u_magnitude_initial_minimal.svg", dpi=400, bbox_inches="tight", transparent=True)
    plt.show()


# Example usage
if __name__ == "__main__":
    trajectories, grid, ts = generate_compressible_flow(
        n_traj=1, N=64, Nt=11, L=5, T=0.1,
        mu=1.0, RT=1.0, noise_level=0.05,
        seed=1, initial_condition="taylor-green"
    )

    u_field = trajectories[0]
    plot_velocity_magnitude_surface_minimal(u_field, L=5, idx=1)
