# ---------------------------------------------------------------------------
# Hopf oscillator: dynamics, trajectories, true coefficients
# ---------------------------------------------------------------------------

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

from scipy.integrate import solve_ivp  # at top of file if not already imported



def hopf(
    t: float,
    u: np.ndarray,
    mu: float = 1.0,
    omega: float = 1.0,
) -> np.ndarray:
    """
    Planar Hopf normal form:

        x_dot = mu x - omega y - (x^2 + y^2) x
        y_dot = omega x + mu y - (x^2 + y^2) y
    """
    x, y = u
    r2 = x**2 + y**2
    return np.array([
        mu * x - omega * y - r2 * x,
        omega * x + mu * y - r2 * y,
    ])


def generate_hopf_trajectory(
    u0: np.ndarray | None = None,
    T: float = 10.0,
    dt: float = 1e-3,
    mu: float = 1.0,
    omega: float = 1.0,
    noise_level: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a single Hopf trajectory.

    Returns
    -------
    t : (N,)
        Time vector.
    U : (N, 2)
        State trajectory (possibly noisy).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, dt)

    if u0 is None:
        u0 = rng.uniform(-2.0, 2.0, size=2)

    sol = solve_ivp(
        hopf,
        (t[0], t[-1]),
        u0,
        t_eval=t,
        args=(mu, omega),
        rtol=1e-12,
        atol=1e-12,
    )
    U = sol.y.T

    if noise_level > 0.0:
        U = U + rng.normal(0.0, noise_level, size=U.shape)

    return t, U


def generate_hopf_dataset(
    n_traj: int = 1,
    T: float = 10.0,
    dt: float = 1e-3,
    noise_level: float = 0.0,
    seed: int = 42,
    mu: float = 1.0,
    omega: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Generate multiple Hopf trajectories (list-of-trajectories format).

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
        u0 = rng.uniform(-2.5, 2.5, size=2)
        t, U = generate_hopf_trajectory(
            u0=u0,
            T=T,
            dt=dt,
            mu=mu,
            omega=omega,
            noise_level=noise_level,
            seed=seed + i,
        )
        trajs.append(U)
        times.append(t)

    return trajs, times[0], times


def build_true_hopf_coefficients(mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    True polynomial coefficient matrix for the Hopf oscillator.

    Basis (degree 3, no bias):
        [x, y, x^2, x y, y^2, x^3, x^2 y, x y^2, y^3]

    Returns
    -------
    C_true : (9, 2)
        Coefficients such that dU/dt = Theta(U) @ C_true.
    """
    C = np.zeros((9, 2))

    # x' = mu x - omega y - (x^2 + y^2) x = mu x - omega y - x^3 - x y^2
    C[0, 0] = mu       # x
    C[1, 0] = -omega   # y
    C[5, 0] = -1.0     # x^3
    C[7, 0] = -1.0     # x y^2

    # y' = omega x + mu y - (x^2 + y^2) y = omega x + mu y - x^2 y - y^3
    C[0, 1] = omega    # x
    C[1, 1] = mu       # y
    C[6, 1] = -1.0     # x^2 y
    C[8, 1] = -1.0     # y^3

    return C

# ---------------------------------------------------------------------------
# Hopf multi-fidelity experiment (HF / LF / MF / MF_w)
# ---------------------------------------------------------------------------
@dataclass
class HopfMFConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the Hopf multi-fidelity SINDy experiment."""

    # multi-fidelity settings
    n_lf: int = 100
    n_hf: int = 10

    # relative noise levels (wrt std of reference trajectory)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # time discretisation
    dt: float = 1e-3
    T_train: float = 10.0
    T_true: float = 10.0

    # Hopf parameters
    mu: float = 1.0
    omega: float = 1.0

    # SINDy settings
    poly_degree: int = 3
    stlsq_threshold: float = 0.5
    n_ensemble_models: int = 100

    # random seeds
    seed_base: int = 0

    # output
    results_filename: str = "hopf_mf_errors.csv"

def _hopf_reference_state_std(cfg: HopfMFConfig) -> float:
    X_ref_list, _, _ = generate_hopf_dataset(
        n_traj=1,
        T=cfg.T_true,
        dt=cfg.dt,
        noise_level=0.0,
        seed=cfg.seed_base,
        mu=cfg.mu,
        omega=cfg.omega,
    )
    return float(np.std(X_ref_list[0]))


def _hopf_batch(
    run_idx: int,
    cfg: HopfMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> MultiTrajectoryGLSData:
    X_hf, t_train, _ = generate_hopf_dataset(
        n_traj=cfg.n_hf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_hf_abs,
        seed=cfg.seed_base + run_idx,
        mu=cfg.mu,
        omega=cfg.omega,
    )
    X_lf, _, _ = generate_hopf_dataset(
        n_traj=cfg.n_lf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_lf_abs,
        seed=cfg.seed_base + 100 + run_idx,
        mu=cfg.mu,
        omega=cfg.omega,
    )
    return MultiTrajectoryGLSData(
        hf=X_hf,
        lf=X_lf,
        t_argument=cfg.dt,
        metadata={"t_grid": t_train},
    )


def _hopf_library(batch: MultiTrajectoryGLSData, cfg: HopfMFConfig):
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )
    return WeakPDELibrary(
        function_library=base_library,
        spatiotemporal_grid=batch.metadata["t_grid"],
    )


def _hopf_true_coefficients(_: MultiTrajectoryGLSData, cfg: HopfMFConfig) -> np.ndarray:
    return build_true_hopf_coefficients(mu=cfg.mu, omega=cfg.omega)


def run_hopf_mf_experiment(
    cfg: HopfMFConfig,
) -> tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    float,
    float,
    float,
]:
    """
    Full Hopf multi-fidelity experiment.

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
        reference_state_std=_hopf_reference_state_std,
        dataset_builder=_hopf_batch,
        library_builder=_hopf_library,
        true_coefficients=_hopf_true_coefficients,
        optimizer_factory=cfg.make_optimizer,
        coef_postprocess=lambda arr: arr.T,
        progress_desc="Monte Carlo Hopf MF",
    )

# ---------------------------------------------------------------------------
# Hopf heteroscedastic GLS experiment (weak / weighted-weak SINDy)
# ---------------------------------------------------------------------------

@dataclass
@dataclass
class HopfGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the heteroscedastic Hopf GLS experiment."""

    n_runs: int = 100

    # time discretisation
    t0: float = 0.0
    t1: float = 10.0
    dt: float = 1e-3

    # Hopf parameters
    mu: float = 1.0
    omega: float = 1.0

    # heteroscedastic noise model: sigma(t) = sigma0 + alpha * |r(t) - r*|
    sigma0: float = 1e-2
    alpha: float = 0.25

    # weak-library settings
    poly_degree: int = 3
    derivative_order: int = 1
    H_xt: float = 0.5
    K: int = 100
    p: int = 2
    include_bias: bool = False

    # SINDy / optimizer settings
    stlsq_threshold: float = 0.5
    n_ensemble_models: int = 20

    # output
    results_filename: str = "hopf_weighted_errors.csv"
    
def _build_hopf_gls_artifacts(
    run_idx: int,
    cfg: HopfGLSConfig,
    rng: np.random.Generator,
) -> IntraTrajectoryGLSData:
    """Construct data/libraries for one Hopf GLS run."""
    T = cfg.t1 - cfg.t0
    t_eval, U_clean = generate_hopf_trajectory(
        u0=rng.uniform(-2.5, 2.5, size=2),
        T=T,
        dt=cfg.dt,
        mu=cfg.mu,
        omega=cfg.omega,
        noise_level=0.0,
        seed=None,
    )

    # Distance from limit cycle r* = sqrt(mu)
    r = np.linalg.norm(U_clean, axis=1)
    r_star = np.sqrt(cfg.mu)
    d = np.abs(r - r_star)

    sigma = cfg.sigma0 + cfg.alpha * d
    variance = sigma**2
    variance = np.maximum(variance, 1e-10)
    std = np.sqrt(variance)

    noise = std[:, None] * rng.standard_normal(size=U_clean.shape)
    U_noisy = U_clean + noise

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
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    # Variance-weighted weak library
    weights_scaled = variance 
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

    # Ones-weighted weak library
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
        t_argument=t_eval,
        libraries=libraries,
        true_coefficients=build_true_hopf_coefficients(mu=cfg.mu, omega=cfg.omega),
    )


def run_hopf_gls_experiment(
    cfg: HopfGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic Hopf GLS experiment.
    """
    rng = np.random.default_rng(cfg.seed_base)

    def builder(run_idx: int, cfg: HopfGLSConfig) -> IntraTrajectoryGLSData:
        return _build_hopf_gls_artifacts(run_idx, cfg, rng)

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="Monte Carlo Hopf GLS",
        coef_postprocess=lambda coef, _method: np.asarray(coef).T,
        coefficient_error_kwargs=lambda _method: {"relative_to_true_support": True},
    )
