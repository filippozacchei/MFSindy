"""
Generate double pendulum trajectories.

Each dataset contains trajectories (angles and velocities)
for the same underlying system with various initial conditions.

Output files:
    ./data/double_pendulum.npz
    ./data/metadata.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# Dynamical system
# -------------------------------
def f(y, g=9.81, L=1.0, m=1.0, c1=0.0, c2=0.0):
    """
    Compute the time derivatives for a planar double pendulum with viscous damping.

    The model describes two rigid links connected by frictional hinges, 
    evolving under gravity. The state vector is:
        y = [θ₁, θ₂, ω₁, ω₂]
    where:
        θ₁ : angle of the first (upper) pendulum w.r.t. the vertical [rad]
        θ₂ : angle of the second (lower) pendulum w.r.t. the vertical [rad]
        ω₁ : angular velocity of the first link [rad/s]
        ω₂ : angular velocity of the second link [rad/s]

    The equations of motion are derived from the Lagrangian of the coupled system
    and include linear damping terms (c₁, c₂). Each rod has equal length (L)
    and mass (m), and gravity acts downward with acceleration g.

    Parameters
    ----------
    y : ndarray of shape (4,)
        Current state vector [θ₁, θ₂, ω₁, ω₂].
    g : float, optional (default=9.81)
        Gravitational acceleration [m/s²].
    L : float, optional (default=1.0)
        Length of each pendulum arm [m].
    m : float, optional (default=1.0)
        Mass of each pendulum bob [kg].
    c1 : float, optional (default=0.07)
        Damping coefficient at the first joint [N·m·s/rad].
    c2 : float, optional (default=0.07)
        Damping coefficient at the second joint [N·m·s/rad].

    Returns
    -------
    dydt : ndarray of shape (4,)
        Time derivatives [dθ₁/dt, dθ₂/dt, dω₁/dt, dω₂/dt].

    Notes
    -----
    The equations are implemented following the canonical form:
        dθ₁/dt = ω₁
        dθ₂/dt = ω₂
        dω₁/dt = F₁(θ₁, θ₂, ω₁, ω₂)
        dω₂/dt = F₂(θ₁, θ₂, ω₁, ω₂)

    The coupling between the two links introduces strong nonlinearity and 
    chaotic dynamics for moderate initial conditions. The damping terms 
    (c₁, c₂) ensure bounded trajectories suitable for system identification.

    Examples
    --------
    >>> y = np.array([np.pi/4, np.pi/6, 0.0, 0.0])
    >>> f(y)
    array([0.0, 0.0, -5.47..., -2.81...])
    """
    th1, th2, w1, w2 = y
    s12 = np.sin(th1 - th2)
    c12 = np.cos(th1 - th2)
    denom = 1 + s12**2

    dth1 = w1
    dth2 = w2

    num1 = (-g*(2*np.sin(th1) + np.sin(th1 - 2*th2))
            - 2*s12*(w2**2 + w1**2*c12))
    dw1 = num1 / (2 * L * denom) - (c1 / (m * L**2)) * w1

    num2 = (2*s12*(2*w1**2 + (g/L)*np.cos(th1) + w2**2*c12)
            - 2*(g/L)*np.sin(th2)*denom)
    dw2 = num2 / (2 * denom) - (c2 / (m * L**2)) * w2

    return np.array([dth1, dth2, dw1, dw2])

def rk4_step(y, h):
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(y0, T=10.0, dt=1/240):
    n = int(T / dt)
    Y = np.zeros((n, len(y0)))
    Y[0] = y0
    for i in range(1, n):
        Y[i] = rk4_step(Y[i-1], dt)
    t = np.arange(n) * dt
    return t, Y

# -------------------------------
# Data generation
# -------------------------------


def generate_dataset(
    noise,
    n_traj=5,
    T=5.0,
    dt=0.001,
    seed=42,
    n_per_trajectory=1,
    bounds: tuple = ((-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2),
                     (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)),
):
    """
    Generate multiple double pendulum trajectories for identification experiments.
    """
    from pyDOE import lhs  # ensures reproducible stratified sampling
    rng = np.random.default_rng(seed)

    # Generate initial conditions using Latin Hypercube Sampling
    dim = 4
    samples = lhs(dim, samples=n_traj)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    y0s = lows + (highs - lows) * samples  # scale LHS samples into bounds
    all = []
    t_all = []
    
    # generate y0 from bouynds y0s

    for y0 in y0s:
        t, Y_true = simulate(y0, T, dt)
        
        # High-fidelity
        for _ in range(n_per_trajectory):
            Y_noisy = Y_true + rng.normal(scale=noise, size=Y_true.shape)
            all.append(Y_noisy)
            t_all.append(t)    

    return all, t, t_all
    
import numpy as np
import matplotlib.pyplot as plt



def generate_double_pendulum_reference_and_fidelities(
    y0,
    T_ref=20.0, dt_ref=1e-3,
    T_short=3.0, dt_hf=1e-3, dt_lf=5e-3,
    noise_hf=0.03, noise_lf=0.15,
    seed=0
):
    rng = np.random.default_rng(seed)

    # Reference long trajectory (smooth background)
    t_ref, Y_ref = simulate(y0, T=T_ref, dt=dt_ref)

    # High fidelity short trajectory
    t_hf, Y_hf = simulate(y0, T=T_short, dt=dt_hf)
    Y_hf_noisy = Y_hf + rng.normal(scale=noise_hf, size=Y_hf.shape)

    # Low fidelity short trajectory (coarser time + larger noise)
    t_lf, Y_lf = simulate(y0, T=T_short, dt=dt_lf)
    Y_lf_noisy = Y_lf + rng.normal(scale=noise_lf, size=Y_lf.shape)

    return (Y_ref, Y_hf_noisy, Y_lf_noisy)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# --- Convert (theta1, theta2) → Cartesian coordinates ---
def to_cartesian(Y, L=1.0):
    th1, th2 = Y[:,0], Y[:,1]
    x1 = L * np.sin(th1)
    y1 = -L * np.cos(th1)
    x2 = x1 + L * np.sin(th2)
    y2 = y1 - L * np.cos(th2)
    return x1, y1, x2, y2

def extract_theta(Y):
    """Return θ1(t) and θ2(t) from state trajectory Y."""
    return Y[:, 0], Y[:, 1]

if __name__ == "__main__":
    # Reference chaotic trajectory (long)
    y0 = np.array([.76241, -.43242413, 0.543214, -1.243214312])
    t_ref, Y_ref = simulate(y0, T=20.0, dt=1/300)

    # HF & LF short trajectories
    t_hf, Y_hf = simulate(y0, T=2.0, dt=1/1000)
    t_lf, Y_lf = simulate(y0, T=2.0, dt=1/250)

    rng = np.random.default_rng()
    Y_hf += rng.normal(scale=0.001, size=Y_hf.shape)
    Y_lf += rng.normal(scale=0.05, size=Y_lf.shape)

    # Convert
    x1_ref, y1_ref, x2_ref, y2_ref = to_cartesian(Y_ref)
    x1_hf, y1_hf, x2_hf, y2_hf = to_cartesian(Y_hf)
    x1_lf, y1_lf, x2_lf, y2_lf = to_cartesian(Y_lf)

    # --- Plot ---
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=350)

    # Background long trajectory
    ax.plot(x2_ref, y2_ref, color="lightgray", lw=0.9, alpha=0.6, zorder=1)

    # Draw the pendulum at the initial condition
    x1_0, y1_0, x2_0, y2_0 = x1_ref[0], y1_ref[0], x2_ref[0], y2_ref[0]
    ax.plot([0, x1_0], [0, y1_0], color="black", lw=2.0, zorder=3)
    ax.plot([x1_0, x2_0], [y1_0, y2_0], color="black", lw=2.0, zorder=3)
    ax.scatter([0, x1_0, x2_0], [0, y1_0, y2_0], color="black", s=28, zorder=4)

    # Low Fidelity (LF): slightly larger, hollow squares
    ax.scatter(x2_lf, y2_lf, marker=".", color="#e31a1c", linewidth=1.2, zorder=5, s=1)
    
    # High Fidelity (HF): clean points, solid circles
    ax.scatter(x2_hf, y2_hf, color="#1f78b4", alpha=0.9, marker=".", zorder=6, s=1)
    
    # Aesthetic cleanup
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("double_pendulum_multifidelity_tip.png",
                dpi=450, bbox_inches="tight", transparent=True)
    
    y0 = np.array([1.3241, -0.143242413, 0.543214, -1.243214312])

    # Reference clean trajectory
    t_ref, Y_ref = simulate(y0, T=2.0, dt=1/1000)
    theta1_ref, theta2_ref = extract_theta(Y_ref)

    # High fidelity (same resolution, small noise)
    t_hf, Y_hf = simulate(y0, T=2.0, dt=1/1000)
    rng = np.random.default_rng()
    Y_hf = Y_hf + rng.normal(scale=0.01, size=Y_hf.shape)
    theta1_hf, theta2_hf = extract_theta(Y_hf)

    # Low fidelity (coarser, noisy)
    t_lf, Y_lf = simulate(y0, T=2.0, dt=1/100)
    Y_lf = Y_lf + rng.normal(scale=0.05, size=Y_lf.shape)
    theta1_lf, theta2_lf = extract_theta(Y_lf)

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=350)

    # Low-fidelity
    ax.plot(t_lf, theta1_lf, linestyle="--", lw=1.8,
            marker="o", markersize=4,
            markerfacecolor="none", markeredgecolor="#e31a1c",
            color="#e31a1c", alpha=0.9)
    ax.plot(t_lf, theta2_lf, linestyle="--", lw=1.8,
            marker="o", markersize=4,
            markerfacecolor="none", markeredgecolor="#e31a1c",
            color="#e31a1c", alpha=0.9, label="Low fidelity")
    
        # High-fidelity
    ax.plot(t_hf, theta1_hf, color="#1f78b4", lw=2.2, alpha=0.95)
    ax.plot(t_hf, theta2_hf, color="#1f78b4", lw=2.2, alpha=0.95, label="High fidelity")

    # Labels
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\theta_1(t),\, \theta_2(t)$")

    # Minimal legend (no frame)
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig("double_pendulum_theta_time_multifidelity.png",
                dpi=450, bbox_inches="tight", transparent=True)
