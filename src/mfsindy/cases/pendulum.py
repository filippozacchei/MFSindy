# %% pendulum_utils.py
"""
Utilities for single-pendulum experiments:
- dynamics and trajectory generators
- true coefficient matrix for linear pendulum model
- generic coefficient error function
- multi-fidelity Hopf-style experiment for the pendulum
- heteroscedastic GLS-style experiment for the pendulum
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

import numpy as np
import pandas as pd

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary

from mfsindy.experiments import (
    EnsembleConfigMixin,
    IntraTrajectoryGLSData,
    MonteCarloConfig,
    MultiTrajectoryGLSData,
    coefficient_errors,
    run_intra_trajectory_gls_experiment,
    run_monte_carlo_experiment,
    run_multi_trajectory_gls_experiment,
)
from mfsindy.weighted_weak_pde_library import WeightedWeakPDELibrary

# ---------------------------------------------------------------------------
# Core single-pendulum dynamics + trajectories
# ---------------------------------------------------------------------------


def pendulum_rhs(
    y: np.ndarray,
    g: float = 9.81,
    L: float = 1.0,
    c: float = 0.1,
) -> np.ndarray:
    """
    Time derivative for a planar single pendulum with viscous damping.

    State:
        y = [theta, omega].

    Equations:
        dtheta/dt = omega
        domega/dt = -(g/L) * theta - c * omega
    """
    theta, omega = y
    dtheta = omega
    domega = -(g / L) * theta - c * omega
    return np.array([dtheta, domega])


def _rk4_step_pendulum(
    y: np.ndarray,
    h: float,
    g: float,
    L: float,
    c: float,
) -> np.ndarray:
    """One RK4 step for the damped pendulum."""
    k1 = pendulum_rhs(y, g=g, L=L, c=c)
    k2 = pendulum_rhs(y + 0.5 * h * k1, g=g, L=L, c=c)
    k3 = pendulum_rhs(y + 0.5 * h * k2, g=g, L=L, c=c)
    k4 = pendulum_rhs(y + h * k3, g=g, L=L, c=c)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_pendulum_trajectory(
    y0: np.ndarray | None = None,
    T: float = 10.0,
    dt: float = 1e-3,
    g: float = 9.81,
    L: float = 1.0,
    c: float = 0.1,
    noise_level: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the single pendulum from initial state y0 up to time T with RK4.

    Parameters
    ----------
    y0 : array-like or None
        Initial condition [theta0, omega0]. If None, drawn from a box.
    T : float
        Final time.
    dt : float
        Time step.
    noise_level : float
        Standard deviation of additive Gaussian noise on [theta, omega].
    seed : int or None
        RNG seed for initial condition and noise.

    Returns
    -------
    t : (N,)
        Time vector.
    Y : (N, 2)
        State trajectory (possibly noisy).
    """
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt

    if y0 is None:
        # Mildly nonlinear initial condition box
        theta0 = rng.uniform(-1.0, 1.0)
        omega0 = rng.uniform(-1.0, 1.0)
        y0 = np.array([theta0, omega0])

    Y = np.zeros((n_steps, 2))
    Y[0] = y0

    for k in range(1, n_steps):
        Y[k] = _rk4_step_pendulum(Y[k - 1], dt, g=g, L=L, c=c)

    if noise_level > 0.0:
        Y += rng.normal(0.0, noise_level, size=Y.shape)

    return t, Y


def generate_pendulum_dataset(
    n_traj: int = 1,
    T: float = 10.0,
    dt: float = 1e-3,
    noise_level: float = 0.0,
    seed: int = 42,
    g: float = 9.81,
    L: float = 1.0,
    c: float = 0.1,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Generate multiple pendulum trajectories (list-of-trajectories format).

    Returns
    -------
    trajs : list of (N, 2)
    t_shared : (N,)
    times : list[(N,)] (all identical)
    """
    rng = np.random.default_rng(seed)

    trajs: list[np.ndarray] = []
    times: list[np.ndarray] = []

    for i in range(n_traj):
        # Vary initial conditions mildly
        theta0 = rng.uniform(-1.0, 1.0)
        omega0 = rng.uniform(-1.0, 1.0)
        y0 = np.array([theta0, omega0])

        t, Y = simulate_pendulum_trajectory(
            y0=y0,
            T=T,
            dt=dt,
            g=g,
            L=L,
            c=c,
            noise_level=noise_level,
            seed=seed + i,
        )
        trajs.append(Y)
        times.append(t)

    return trajs, times[0], times


def build_true_pendulum_coefficients(
    g: float = 9.81,
    L: float = 1.0,
    c: float = 0.1,
) -> np.ndarray:
    """
    True coefficient matrix for the *linear* pendulum model
    in the polynomial basis [theta, omega].

    Basis ordering (no bias):
        [theta, omega].

    Model:
        dtheta = 0 * theta + 1 * omega
        domega = -(g/L) * theta - c * omega

    Returns
    -------
    C_true : (2, 2)
        Coefficients such that dY/dt = Theta(Y) @ C_true.
        Rows correspond to [theta, omega], columns to [dtheta, domega].
    """
    C = np.zeros((2, 2))

    # dtheta/dt
    C[0, 0] = 0.0        # theta
    C[1, 0] = 1.0        # omega

    # domega/dt
    C[0, 1] = -(g / L)   # theta
    C[1, 1] = -c         # omega

    return C


# ---------------------------------------------------------------------------
# Pendulum multi-fidelity experiment (HF / LF / MF / MF_w)
# ---------------------------------------------------------------------------


@dataclass
@dataclass
class PendulumMultiTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the pendulum multi-fidelity SINDy experiment."""

    # multi-fidelity settings
    n_lf: int = 10
    n_hf: int = 1

    # relative noise levels (wrt std of reference trajectory)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # time discretisation
    dt: float = 1e-3
    T_train: float = 5.0
    T_true: float = 10.0

    # physical parameters
    g: float = 9.81
    L: float = 1.0
    c: float = 0.1

    # SINDy settings
    poly_degree: int = 1
    stlsq_threshold: float = 0.01
    n_ensemble_models: int = 100

    # random seeds
    seed_base: int = 0

    # output
    results_filename: str = "pendulum_mf_errors.csv"


def _pendulum_reference_state_std(cfg: PendulumMultiTrajectoryGLSConfig) -> float:
    X_ref_list, _, _ = generate_pendulum_dataset(
        n_traj=1,
        T=cfg.T_true,
        dt=cfg.dt,
        noise_level=0.0,
        seed=cfg.seed_base,
        g=cfg.g,
        L=cfg.L,
        c=cfg.c,
    )
    return float(np.std(X_ref_list[0]))


def _pendulum_batch(
    run_idx: int,
    cfg: PendulumMultiTrajectoryGLSConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> MultiTrajectoryGLSData:
    X_hf, t_train, _ = generate_pendulum_dataset(
        n_traj=cfg.n_hf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_hf_abs,
        seed=cfg.seed_base + run_idx,
        g=cfg.g,
        L=cfg.L,
        c=cfg.c,
    )
    X_lf, _, _ = generate_pendulum_dataset(
        n_traj=cfg.n_lf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_lf_abs,
        seed=cfg.seed_base + 100 + run_idx,
        g=cfg.g,
        L=cfg.L,
        c=cfg.c,
    )
    return MultiTrajectoryGLSData(
        hf=X_hf,
        lf=X_lf,
        t_argument=cfg.dt,
        metadata={"t_grid": t_train},
    )


def _pendulum_library(batch: MultiTrajectoryGLSData, cfg: PendulumMultiTrajectoryGLSConfig):
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )
    return WeakPDELibrary(
        function_library=base_library,
        spatiotemporal_grid=batch.metadata["t_grid"],
    )


def _pendulum_true_coefficients(_: MultiTrajectoryGLSData, cfg: PendulumMultiTrajectoryGLSConfig) -> np.ndarray:
    return build_true_pendulum_coefficients(g=cfg.g, L=cfg.L, c=cfg.c)


def run_pendulum_multi_trajectory_gls_experiment(
    cfg: PendulumMultiTrajectoryGLSConfig,
) -> tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    float,
    float,
    float,
]:
    """
    Full pendulum multi-fidelity experiment.

    Returns
    -------
    df_errors    : long-format DataFrame (run, model, metric, value)
    mae_errors   : dict[model] -> array of MAE errors
    l0_errors    : dict[model] -> array of L0 errors
    state_std    : reference state standard deviation
    noise_hf_abs : absolute HF noise level
    noise_lf_abs : absolute LF noise level
    """
    return run_multi_trajectory_gls_experiment(
        cfg,
        reference_state_std=_pendulum_reference_state_std,
        dataset_builder=_pendulum_batch,
        library_builder=_pendulum_library,
        true_coefficients=_pendulum_true_coefficients,
        optimizer_factory=cfg.make_optimizer,
        coef_postprocess=lambda arr: arr.T,
        progress_desc="Monte Carlo pendulum MF",
    )


# ---------------------------------------------------------------------------
# Pendulum heteroscedastic GLS experiment (weak / weighted-weak SINDy)
# ---------------------------------------------------------------------------
@dataclass
class PendulumIntraTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the heteroscedastic pendulum GLS experiment."""

    # time discretisation
    t0: float = 0.0
    t1: float = 10.0
    dt: float = 1e-3

    # physical parameters
    g: float = 9.81
    L: float = 1.0
    c: float = 0.1

    # heteroscedastic noise model: sigma(t) = sigma0 + alpha * |omega(t)|
    sigma0: float = 0.0
    alpha: float = 0.15

    # weak-library settings
    poly_degree: int = 1
    derivative_order: int = 1
    H_xt: float = 0.1
    K: int = 500
    p: int = 2
    include_bias: bool = False

    # SINDy / optimizer settings
    stlsq_threshold: float = 0.01
    n_ensemble_models: int = 100

    results_filename: str = "pendulum_weighted_errors.csv"


def _build_pendulum_gls_artifacts(
    run_idx: int,
    cfg: PendulumIntraTrajectoryGLSConfig,
    rng: np.random.Generator,
) -> IntraTrajectoryGLSData:
    """Build noisy trajectory and weak libraries for a pendulum GLS run."""
    T = cfg.t1 - cfg.t0

    # Sample initial condition from a box
    theta0 = rng.uniform(-1.0, 1.0)
    omega0 = rng.uniform(-0.5, 0.5)
    y0 = np.array([theta0, omega0])

    t_eval, Y_clean = simulate_pendulum_trajectory(
        y0=y0,
        T=T,
        dt=cfg.dt,
        g=cfg.g,
        L=cfg.L,
        c=cfg.c,
        noise_level=0.0,
        seed=None,
    )

    # Heteroscedastic noise: sigma(t) depends on |omega(t)|
    omega_mag = np.abs(Y_clean[:, 0])  # column 1 = omega
    sigma = cfg.sigma0 + cfg.alpha * omega_mag
    variance = np.maximum(sigma**2, 1e-10)
    std = np.sqrt(variance)

    noise = std[:, None] * rng.standard_normal(size=Y_clean.shape)
    Y_noisy = Y_clean + noise

    # Spatiotemporal grid and polynomial library
    XT = t_eval[:, None]
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=cfg.include_bias,
    )

    tf_seed = cfg.seed_base + 1000 + run_idx

    # Unweighted weak library
    np.random.seed(tf_seed)
    weak_lib = WeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    # Variance-weighted weak library
    np.random.seed(tf_seed)
    weighted_weak_lib_var = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        spatiotemporal_weights=variance,
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    # Ones-weighted weak library
    np.random.seed(tf_seed)
    weighted_weak_lib_ones = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        spatiotemporal_weights=np.ones_like(variance),
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    libraries = {
        "No weighting": weak_lib,
        "Variance GLS": weighted_weak_lib_var,
        "Ones GLS": weighted_weak_lib_ones,
    }

    return IntraTrajectoryGLSData(
        data=Y_noisy,
        t_argument=t_eval,
        libraries=libraries,
        true_coefficients=build_true_pendulum_coefficients(g=cfg.g, L=cfg.L, c=cfg.c),
    )


def run_pendulum_intra_trajectory_gls_experiment(
    cfg: PendulumIntraTrajectoryGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic pendulum GLS experiment.
    """
    rng = np.random.default_rng(cfg.seed_base)

    def builder(run_idx: int, cfg: PendulumIntraTrajectoryGLSConfig) -> IntraTrajectoryGLSData:
        return _build_pendulum_gls_artifacts(run_idx, cfg, rng)

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="Monte Carlo pendulum GLS",
        coef_postprocess=lambda coef, _method: np.asarray(coef).T,
        coefficient_error_kwargs=lambda _method: {"relative_to_true_support": True},
    )
