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
    
if __name__ == "__main__":
    # Parameters
    rng = np.random.default_rng(0)
    n_ic = 125
    n_lf = 10
    n_hf = 1
    noise_lf = 0.5
    noise_hf = 0.05
    T = 5.0
    dt = 1 / 1000

    # Random initial conditions: [θ₁, θ₂, ω₁, ω₂]
    y0s = [
        rng.uniform(low=[-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2], high=[np.pi/2, np.pi/2, np.pi/2, np.pi/2])
        for _ in range(n_ic)
    ]

    # Generate dataset
    data = generate_dataset(
        y0s=y0s,
        n_hf=n_hf,
        n_lf=n_lf,
        noise_lf=noise_lf,
        noise_hf=noise_hf,
        T=T,
        dt=dt,
        out_path="./data/double_pendulum_dataset.npz",
    )

    # Save metadata for reproducibility
    meta = dict(
        n_ic=n_ic,
        n_hf=n_hf,
        n_lf=n_lf,
        noise_hf=noise_hf,
        noise_lf=noise_lf,
        T=T,
        dt=dt,
        seed=int(rng.bit_generator.state["state"]["state"]),
    )
    Path("./data").mkdir(exist_ok=True)
    with open("./data/metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
    print("Saved metadata → ./data/metadata.json")
