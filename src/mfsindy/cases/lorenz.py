# ---------------------------------------------------------------------------
# Lorenz system: generators and true coefficients
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary

from mfsindy.cases.common import (
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



def lorenz(
    t: float,
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> list[float]:
    """Standard Lorenz system."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def generate_lorenz_trajectory(
    y0: np.ndarray | None = None,
    T: float = 10.0,
    dt: float = 1e-3,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    noise_level: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single Lorenz trajectory and its derivatives.

    Returns
    -------
    t : (N,)
        Time vector.
    X : (N, 3)
        State trajectory (possibly noisy).
    Xdot : (N, 3)
        True derivatives (noise-free).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, dt)

    if y0 is None:
        y0 = rng.uniform([-20.0, -20.0, 20.0], [20.0, 20.0, 30.0])

    sol = solve_ivp(
        lorenz,
        (t[0], t[-1]),
        y0,
        t_eval=t,
        args=(sigma, rho, beta),
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )

    X = sol.y.T
    Xdot = np.array([lorenz(ti, xi, sigma, rho, beta) for ti, xi in zip(sol.t, X)])

    if noise_level > 0:
        X += rng.normal(0.0, noise_level, size=X.shape)

    return t, X, Xdot


def generate_lorenz_dataset(
    n_traj: int = 1,
    T: float = 10.0,
    dt: float = 1e-3,
    noise_level: float = 0.0,
    seed: int = 42,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Generate multiple Lorenz trajectories (list-of-trajectories format).

    Returns
    -------
    trajs : list of (N, 3)
    t_shared : (N,)
    times : list of (N,) (all identical)
    """
    rng = np.random.default_rng(seed)

    trajs: list[np.ndarray] = []
    derivs: list[np.ndarray] = []
    times: list[np.ndarray] = []

    for i in range(n_traj):
        y0 = rng.uniform([-10.0, -10.0, 20.0], [10.0, 10.0, 30.0])
        t, X, Xdot = generate_lorenz_trajectory(
            y0=y0,
            T=T,
            dt=dt,
            sigma=sigma,
            rho=rho,
            beta=beta,
            noise_level=noise_level,
            seed=seed + i,
        )
        trajs.append(X)
        derivs.append(Xdot)
        times.append(t)

    return trajs, times[0], times


def build_true_coefficient_matrix() -> np.ndarray:
    """
    True polynomial coefficient matrix for the Lorenz system.

    Polynomial terms (no bias):
        [x, y, z, x^2, x y, x z, y^2, y z, z^2]

    Returns
    -------
    C : (9, 3)
        Coefficient matrix such that dX/dt = Theta(X) @ C.
    """
    C = np.zeros((9, 3))

    # dx/dt = -10 x + 10 y
    C[0, 0] = -10.0  # x
    C[1, 0] = 10.0   # y

    # dy/dt = 28 x - x z - y
    C[0, 1] = 28.0   # x
    C[5, 1] = -1.0   # x z
    C[1, 1] = -1.0   # y

    # dz/dt = x y - (8/3) z
    C[4, 2] = 1.0          # x y
    C[2, 2] = -8.0 / 3.0   # z

    return C



# ---------------------------------------------------------------------------
# Lorenz multi-fidelity experiment (HF / LF / MF / MF_w)
# ---------------------------------------------------------------------------

@dataclass
class LorenzMFConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the Lorenz multi-fidelity SINDy experiment."""

    # multi-fidelity settings
    n_lf: int = 100
    n_hf: int = 10

    # relative noise levels (wrt std of reference trajectory)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # time discretization
    dt: float = 1e-3
    T_train: float = 0.1
    T_true: float = 100.0
    T_forecast: float = 2.0

    # SINDy settings
    poly_degree: int = 2
    stlsq_threshold: float = 0.5
    n_ensemble_models: int = 200

    # random seeds
    seed_base: int = 231
    seed_forecast_ic: int = 999

    # output
    results_filename: str = "lorenz_mf_errors.csv"


def _lorenz_reference_state_std(cfg: LorenzMFConfig) -> float:
    """Reference standard deviation used to scale HF/LF noise levels."""

    X_true_list, _, _ = generate_lorenz_dataset(
        n_traj=1,
        T=cfg.T_true,
        dt=cfg.dt,
        noise_level=0.0,
        seed=cfg.seed_base,
    )
    return float(np.std(X_true_list[0]))


def _lorenz_batch(
    run_idx: int,
    cfg: LorenzMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> MultiTrajectoryGLSData:
    """Build the HF/LF training batch for a single Monte Carlo run."""

    X_hf, t_train, _ = generate_lorenz_dataset(
        n_traj=cfg.n_hf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_hf_abs,
        seed=cfg.seed_base + run_idx,
    )
    X_lf, _, _ = generate_lorenz_dataset(
        n_traj=cfg.n_lf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_lf_abs,
        seed=cfg.seed_base + 100 + run_idx,
    )
    return MultiTrajectoryGLSData(
        hf=X_hf,
        lf=X_lf,
        t_argument=cfg.dt,
        metadata={"t_grid": t_train},
    )


def _lorenz_library(batch: MultiTrajectoryGLSData, cfg: LorenzMFConfig):
    """Shared weak-form library for all fidelity variants."""

    poly_lib = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )
    return WeakPDELibrary(
        function_library=poly_lib,
        spatiotemporal_grid=batch.metadata["t_grid"],
    )


def _lorenz_true_coefficients(_: MultiTrajectoryGLSData, cfg: LorenzMFConfig) -> np.ndarray:
    return build_true_coefficient_matrix()


def run_lorenz_mf_experiment(
    cfg: LorenzMFConfig,
) -> tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    float,
    float,
    float,
]:
    """
    Full Lorenz multi-fidelity experiment.

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
        reference_state_std=_lorenz_reference_state_std,
        dataset_builder=_lorenz_batch,
        library_builder=_lorenz_library,
        true_coefficients=_lorenz_true_coefficients,
        optimizer_factory=cfg.make_optimizer,
        coef_postprocess=lambda arr: arr.T,
        progress_desc="Monte Carlo Lorenz MF",
    )

# ---------------------------------------------------------------------------
# Lorenz heteroscedastic GLS experiment (weak / weighted-weak SINDy)
# ---------------------------------------------------------------------------

@dataclass
class LorenzGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the heteroscedastic Lorenz GLS experiment."""

    # time discretisation
    t0: float = 0.0
    t1: float = 10.0
    dt: float = 1e-3

    # heteroscedastic noise level (variance ∝ alpha^2 * ||U||^2)
    noise_level: float = 0.25

    # weak-library settings
    poly_degree: int = 2
    derivative_order: int = 1
    H_xt: float = 0.01
    K: int = int(5 * (t1-t0) / H_xt)          # can be set as int(5 * (t1-t0) / H_xt)
    p: int = 2
    include_bias: bool = False

    # SINDy / optimizer settings
    stlsq_threshold: float = 0.5
    n_ensemble_models: int = 100

    # random seeds
    seed_base: int = 0

    # output
    results_filename: str = "lorenz_weighted_errors.csv"


def _make_lorenz_gls_artifacts(
    run_idx: int,
    cfg: LorenzGLSConfig,
    rng: np.random.Generator,
) -> IntraTrajectoryGLSData:
    """Build data and libraries for one Lorenz GLS run."""
    # 1) Clean trajectory from random initial condition
    u0 = rng.uniform(-20.0, 20.0, size=3)
    T = cfg.t1 - cfg.t0
    t_eval, U_clean, _ = generate_lorenz_trajectory(
        y0=u0,
        T=T,
        dt=cfg.dt,
        noise_level=0.0,
        seed=None,
    )

    # 2) Heteroscedastic noise: variance ∝ (alpha * ||U||)^2
    alpha = cfg.noise_level
    d = np.linalg.norm(U_clean, axis=1)
    variance = (alpha * d) ** 2
    variance = np.maximum(variance, 1e-8)
    std = np.sqrt(variance)
    noise = std[:, None] * rng.standard_normal(size=U_clean.shape)
    U_noisy = U_clean + noise

    # 3) Spatiotemporal grid and polynomial library
    XT = t_eval[:, None]
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=cfg.include_bias,
    )

    tf_seed = cfg.seed_base + 1000 + run_idx

    # 3a) Unweighted weak library
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

    # 3b) Weighted-by-variance weak library
    weights_scaled = variance / np.mean(variance)
    np.random.seed(tf_seed)
    weighted_weak_lib_var = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        spatiotemporal_weights=weights_scaled,
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    # 3c) Weighted-by-ones weak library
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

    C_true = build_true_coefficient_matrix()

    libraries = {
        "No weighting": weak_lib,
        "Variance GLS": weighted_weak_lib_var,
        "Ones GLS": weighted_weak_lib_ones,
    }

    return IntraTrajectoryGLSData(
        data=U_noisy,
        t_argument=t_eval,
        libraries=libraries,
        true_coefficients=C_true,
    )


def run_lorenz_gls_experiment(
    cfg: LorenzGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic Lorenz GLS experiment.
    """
    rng = np.random.default_rng(cfg.seed_base)

    def builder(run_idx: int, cfg: LorenzGLSConfig) -> IntraTrajectoryGLSData:
        return _make_lorenz_gls_artifacts(run_idx, cfg, rng)

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="Monte Carlo Lorenz GLS",
        coef_postprocess=lambda coef, _method: np.asarray(coef).T,
        coefficient_error_kwargs=lambda _method: {"relative_to_true_support": True},
    )
