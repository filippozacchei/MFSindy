# utils.py
"""
Utilities for 1D viscous Burgers experiments:

- Finite-difference Burgers solver (periodic BC)
- Random Gaussian-bump initial conditions
- Heteroscedastic noise based on |u_x|
- Weak / weighted-weak libraries (WeakPDELibrary, WeightedWeakPDELibrary)
- Monte Carlo GLS experiment (heteroscedastic noise)
- Multi-fidelity data generation and PDE-SINDy models (HF / LF / MF / MF_w)
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
    fit_intra_trajectory_gls_models,
    run_intra_trajectory_gls_experiment,
    run_monte_carlo_experiment,
    run_multi_trajectory_gls_experiment,
)
from mfsindy.weighted_weak_pde_library import WeightedWeakPDELibrary


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BurgersConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the 1D Burgers GLS experiment (heteroscedastic noise)."""

    n_runs: int = 100

    # Domain and discretisation
    L: float = 8.0          # half-domain, domain [-L, L]
    NX: int = 256           # number of spatial points
    t0: float = 0.0
    t1: float = 10.0
    dt: float = 1e-2
    nu: float = 0.1         # viscosity

    # Heteroscedastic noise (variance ∝ (alpha * |u_x|)^2)
    noise_level: float = 0.25

    # Weak-library settings
    poly_degree: int = 1
    derivative_order: int = 2
    H_xt: float | None = None  # test-function support; None → library default
    K: int = 100               # number of weak test functions
    include_bias: bool = True

    # SINDy / optimizer settings
    stlsq_threshold: float = 0.05
    n_ensemble_models: int = 100

    # Output
    results_filename: str = "burgers_weighted_errors.csv"


@dataclass
class BurgersMultiTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration container for the Burgers multi-fidelity experiment."""

    # spatial / temporal discretization
    L: float = 8.0
    NX: int = 256
    dt: float = 1e-2
    T_train: float = 10.0
    nu: float = 0.1

    # multi-fidelity settings (numbers of trajectories)
    n_lf: int = 100
    n_hf: int = 10
    H_xt: float | None = None
    K: int = 100

    # relative noise levels (wrt state std)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # SINDy / PDE-library settings
    poly_degree: int = 1               # polynomial degree in u
    derivative_order: int = 2          # spatial derivative order
    include_interaction: bool = True
    include_bias: bool = True

    stlsq_threshold: float = 0.05
    n_ensemble_models: int = 200

    # random seeds
    seed_base: int = 231

    # output
    results_filename: str = "burgers_mf_errors.csv"


# ---------------------------------------------------------------------------
# Basic grids and solver
# ---------------------------------------------------------------------------

def make_space_time_grid(cfg: BurgersConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial and temporal grids."""
    x = np.linspace(-cfg.L, cfg.L, cfg.NX, endpoint=False)
    t = np.arange(cfg.t0, cfg.t1, cfg.dt)
    return x, t


def burgers_solver(u0: np.ndarray, cfg: BurgersConfig) -> np.ndarray:
    """
    Explicit finite-difference solver for 1D viscous Burgers with periodic BCs.

        u_t + u u_x = nu u_xx

    Parameters
    ----------
    u0 : (NX,)
        Initial condition at t = t0.

    Returns
    -------
    U : (NT, NX)
        Time snapshots, with time along axis 0.
    """
    x, t = make_space_time_grid(cfg)
    dx = x[1] - x[0]
    NT = len(t)
    NX = u0.shape[0]

    u = u0.copy()
    U = np.zeros((NT, NX))
    U[0] = u

    for n in range(1, NT):
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
        uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
        u = u + cfg.dt * (-u * ux + cfg.nu * uxx)
        U[n] = u

    return U


def random_initial_condition(
    rng: np.random.Generator,
    cfg: BurgersConfig,
) -> np.ndarray:
    """
    Random Gaussian bump initial condition:

        u0(x) = amp * exp(-(x - center)^2 / (2 * width^2)).
    """
    x, _ = make_space_time_grid(cfg)
    amp = rng.uniform(0.3, 1.0)
    center = rng.uniform(-cfg.L / 2.0, cfg.L / 2.0)
    width = rng.uniform(0.5, 1.5)
    return amp * np.exp(-(x - center) ** 2 / (2.0 * width ** 2))


# ---------------------------------------------------------------------------
# True coefficients and error metrics
# ---------------------------------------------------------------------------

def build_true_burgers_coefficients(nu: float) -> np.ndarray:
    """
    True coefficients for Burgers in a 1D polynomial PDE library.

    Assumed feature ordering:
        [1, u, u_x, u_xx, u u_x, u u_xx]

    PDE:
        u_t = 0 * 1
            + 0 * u
            + 0 * u_x
            + nu * u_xx
            - 1 * (u u_x)
            + 0 * (u u_xx).
    """
    C_true = np.zeros((1, 6))
    C_true[0, 3] = nu    # u_xx
    C_true[0, 4] = -1.0  # u u_x
    return C_true



# ---------------------------------------------------------------------------
# Single Monte Carlo run (heteroscedastic GLS experiment)
# ---------------------------------------------------------------------------

def _build_burgers_gls_artifacts(
    run_idx: int,
    cfg: BurgersConfig,
    rng: np.random.Generator,
) -> IntraTrajectoryGLSData:
    """Construct data + weak libraries for a single Burgers GLS run."""
    x, t = make_space_time_grid(cfg)
    dx = x[1] - x[0]

    # 1) Random initial condition and clean trajectory
    u0 = random_initial_condition(rng, cfg)
    U_clean = burgers_solver(u0, cfg).T      # (NT, NX) → transpose
    U = U_clean[:, :, None]                  # (NT, NX, 1) for PySINDy

    # 2) Heteroscedastic noise based on |u_x|
    Ux = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2.0 * dx)
    grad_mag = np.abs(Ux[:, :, 0])           # (NT, NX)

    alpha = cfg.noise_level
    variance = (alpha * grad_mag) ** 2
    variance = np.maximum(variance, 1e-8)
    std = np.sqrt(variance)

    noise = std[:, :, None] * rng.standard_normal(size=U.shape)
    U_noisy = U + noise

    # 3) Spatiotemporal grid (t, x) → shape (NT, NX, 2)
    X, T = np.meshgrid(x, t)
    XT = np.asarray([X, T]).T                # (NT, NX, 2)

    # 4) Base polynomial library in u
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )

    tf_seed = cfg.seed_base + 1000 + run_idx

    # 4a) Unweighted weak library
    np.random.seed(tf_seed)
    weak_lib = WeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        is_uniform=True,
        K=cfg.K,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    # 4b) Variance-weighted weak library
    weights_scaled = variance / np.mean(variance)
    np.random.seed(tf_seed)
    weighted_weak_lib_var = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        spatiotemporal_weights=weights_scaled,
        is_uniform=True,
        K=cfg.K,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    # 4c) Ones-weighted weak library (GLS with unit weights)
    np.random.seed(tf_seed)
    weighted_weak_lib_ones = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        spatiotemporal_weights=np.ones_like(variance),
        is_uniform=True,
        K=cfg.K,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    libraries = {
        "No weighting": weak_lib,
        "Variance GLS": weighted_weak_lib_var,
        "Ones GLS": weighted_weak_lib_ones,
    }

    return IntraTrajectoryGLSData(
        data=U_noisy,
        t_argument=t,
        libraries=libraries,
        true_coefficients=build_true_burgers_coefficients(cfg.nu),
    )


def run_burgers_experiment(
    cfg: BurgersConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic GLS experiment.
    """
    rng = np.random.default_rng(cfg.seed_base)

    def builder(run_idx: int, cfg: BurgersConfig) -> IntraTrajectoryGLSData:
        return _build_burgers_gls_artifacts(run_idx, cfg, rng)

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="Monte Carlo Burgers GLS",
    )


# ---------------------------------------------------------------------------
# Multi-fidelity Burgers helpers (LF / HF / MF / MF_w)
# ---------------------------------------------------------------------------

def generate_burgers_dataset(
    n_traj: int,
    T: float,
    dt: float,
    L: float,
    NX: int,
    nu: float,
    noise_level: float,
    seed: int,
):
    """
    Generate a list of noisy Burgers trajectories.

    Each trajectory is a solution U(t, x) with additive i.i.d. Gaussian noise
    with standard deviation = noise_level.

    Returns
    -------
    X_list : list of arrays (Nt, Nx, 1)
        Noisy trajectories.
    t      : (Nt,) time grid (shared).
    x      : (Nx,) spatial grid (shared).
    nu     : float, passed through.
    """
    cfg_tmp = BurgersConfig(L=L, NX=NX, t0=0.0, t1=T, dt=dt, nu=nu)
    x, t = make_space_time_grid(cfg_tmp)
    rng = np.random.default_rng(seed)

    X_list = []
    for _ in range(n_traj):
        u0 = random_initial_condition(rng, cfg_tmp)
        U_clean = burgers_solver(u0, cfg_tmp).T      # (Nt, Nx)
        U = U_clean                                  # (Nt, Nx)

        noise = noise_level * rng.standard_normal(size=U.shape)
        U_noisy = U + noise
        X_list.append(U_noisy[:, :, None])           # (Nt, Nx, 1)

    return X_list, t, x, nu


def _burgers_reference_state_std(cfg: BurgersMultiTrajectoryGLSConfig) -> float:
    """Reference state std used to convert relative noise to absolute values."""

    X_ref_list, _, _, _ = generate_burgers_dataset(
        n_traj=1,
        T=cfg.T_train,
        dt=cfg.dt,
        L=cfg.L,
        NX=cfg.NX,
        nu=cfg.nu,
        noise_level=0.0,
        seed=cfg.seed_base,
    )
    return float(np.std(X_ref_list[0]))


def _burgers_batch(
    run_idx: int,
    cfg: BurgersMultiTrajectoryGLSConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> MultiTrajectoryGLSData:
    """Return HF/LF training trajectories for a single Monte Carlo run."""

    X_hf, t_train, x_grid, _ = generate_burgers_dataset(
        n_traj=cfg.n_hf,
        T=cfg.T_train,
        dt=cfg.dt,
        L=cfg.L,
        NX=cfg.NX,
        nu=cfg.nu,
        noise_level=noise_hf_abs,
        seed=cfg.seed_base + run_idx,
    )
    X_lf, _, _, _ = generate_burgers_dataset(
        n_traj=cfg.n_lf,
        T=cfg.T_train,
        dt=cfg.dt,
        L=cfg.L,
        NX=cfg.NX,
        nu=cfg.nu,
        noise_level=noise_lf_abs,
        seed=cfg.seed_base + 100 + run_idx,
    )
    return MultiTrajectoryGLSData(
        hf=X_hf,
        lf=X_lf,
        t_argument=cfg.dt,
        metadata={"t": t_train, "x": x_grid},
    )


def _burgers_library(batch: MultiTrajectoryGLSData, cfg: BurgersMultiTrajectoryGLSConfig):
    """Shared weak-form Burgers library."""

    x = batch.metadata["x"]
    t = batch.metadata["t"]
    X, T = np.meshgrid(x, t)
    XT = np.asarray([X, T]).T

    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )
    return WeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        is_uniform=True,
        K=cfg.K,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )


def _burgers_true_coefficients(_: MultiTrajectoryGLSData, cfg: BurgersMultiTrajectoryGLSConfig) -> np.ndarray:
    return build_true_burgers_coefficients(cfg.nu)


def run_burgers_multi_trajectory_gls_experiment(
    cfg: BurgersMultiTrajectoryGLSConfig,
) -> tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    float,
    float,
    float,
]:
    """
    Full multi-fidelity Burgers experiment.

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
        reference_state_std=_burgers_reference_state_std,
        dataset_builder=_burgers_batch,
        library_builder=_burgers_library,
        true_coefficients=_burgers_true_coefficients,
        optimizer_factory=cfg.make_optimizer,
        progress_desc="Monte Carlo Burgers MF",
    )

# ---------------------------------------------------------------------------
# Helper: get one set of GLS coefficients for animation / forecasting
# ---------------------------------------------------------------------------

def get_burgers_gls_coefficients(
    cfg: BurgersConfig,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ONE heteroscedastic Burgers dataset and fit the three GLS
    models (no weighting / variance GLS / ones GLS).

    Returns 1D coefficient vectors:

        C_true, C_std, C_var, C_ones

    where
        - C_true : analytic Burgers coefficients (same library ordering)
        - C_std  : WeakPDELibrary (no weighting)
        - C_var  : WeightedWeakPDELibrary with variance weights
        - C_ones : WeightedWeakPDELibrary with all-ones weights
    """
    rng_seed = cfg.seed_base if seed is None else seed
    rng = np.random.default_rng(rng_seed)
    artifacts = _build_burgers_gls_artifacts(
        run_idx=0,
        cfg=cfg,
        rng=rng,
    )

    methods = ["No weighting", "Variance GLS", "Ones GLS"]
    coef_map = fit_intra_trajectory_gls_models(cfg, artifacts, methods)

    C_true = artifacts.true_coefficients.ravel()
    C_std = coef_map["No weighting"].ravel()
    C_var = coef_map["Variance GLS"].ravel()
    C_ones = coef_map["Ones GLS"].ravel()

    return C_true, C_std, C_var, C_ones
