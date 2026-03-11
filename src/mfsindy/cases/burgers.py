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

from mfsindy.cases.common import coefficient_errors, run_monte_carlo_experiment
from mfsindy.weighted_weak_pde_library import WeightedWeakPDELibrary


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BurgersConfig:
    """Configuration for the 1D Burgers GLS experiment (heteroscedastic noise)."""

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

    # Monte Carlo
    n_runs: int = 100

    # Random seeds
    seed_base: int = 0

    # Output
    results_dir: str = "results"
    results_filename: str = "burgers_weighted_errors.csv"


@dataclass
class BurgersMFConfig:
    """Configuration container for the Burgers multi-fidelity experiment."""

    # Monte Carlo
    n_runs: int = 25

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
    results_dir: str = "results"
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

def _run_single_gls_run(
    run_idx: int,
    cfg: BurgersConfig,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the heteroscedastic Burgers GLS experiment.

    Returns
    -------
    errors : dict[str, (L1_error, L0_error)]
        Keys: "No weighting", "Variance GLS", "Ones GLS".
    """
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

    # 5) Ensemble optimizers

    def make_optimizer() -> ps.EnsembleOptimizer:
        return ps.EnsembleOptimizer(
            ps.STLSQ(threshold=cfg.stlsq_threshold),
            bagging=True,
            n_models=cfg.n_ensemble_models,
        )

    opt_std = make_optimizer()
    opt_var = make_optimizer()
    opt_ones = make_optimizer()

    model_std = ps.SINDy(feature_library=weak_lib, optimizer=opt_std)
    model_var = ps.SINDy(feature_library=weighted_weak_lib_var, optimizer=opt_var)
    model_ones = ps.SINDy(feature_library=weighted_weak_lib_ones, optimizer=opt_ones)

    model_std.fit(U_noisy, t=t)
    model_var.fit(U_noisy, t=t)
    model_ones.fit(U_noisy, t=t)

    # 6) True coefficients (fixed feature ordering)
    C_true = build_true_burgers_coefficients(cfg.nu)

    C_std = np.array(model_std.optimizer.coef_)
    C_var = np.array(model_var.optimizer.coef_)
    C_ones = np.array(model_ones.optimizer.coef_)

    # L1 on true support + L0 mismatch
    L1_std, L0_std = coefficient_errors(
        C_std, C_true, relative_to_true_support=True
    )
    L1_var, L0_var = coefficient_errors(
        C_var, C_true, relative_to_true_support=True
    )
    L1_ones, L0_ones = coefficient_errors(
        C_ones, C_true, relative_to_true_support=True
    )

    errors = {
        "No weighting": (L1_std, L0_std),
        "Variance GLS": (L1_var, L0_var),
        "Ones GLS": (L1_ones, L0_ones),
    }
    return errors


def run_burgers_experiment(
    cfg: BurgersConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic GLS experiment.

    Returns
    -------
    df_errors  : long-format DataFrame (run, method, metric, value)
    L1_errors  : dict[method] -> array of L1 errors
    L0_errors  : dict[method] -> array of L0 errors
    """
    methods = ["No weighting", "Variance GLS", "Ones GLS"]
    rng = np.random.default_rng(cfg.seed_base)

    def single_run(run_idx: int):
        return _run_single_gls_run(run_idx=run_idx, cfg=cfg, rng=rng)

    df_errors, L1_errors, L0_errors = run_monte_carlo_experiment(
        n_runs=cfg.n_runs,
        methods=methods,
        single_run_fn=single_run,
        results_dir=cfg.results_dir,
        results_filename=cfg.results_filename,
        metric1_name="L1",
        metric2_name="L0",
        source_col="method",
        progress_desc="Monte Carlo Burgers GLS",
    )

    return df_errors, L1_errors, L0_errors


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


def build_ensemble_sindy_models_burgers(
    X_hf,
    X_lf,
    t,
    x,
    cfg: BurgersMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
):
    """
    Build four ensemble PDE-SINDy models for 1D Burgers:

    - HF   : trained on high-fidelity trajectories only
    - LF   : trained on low-fidelity trajectories only
    - MF   : trained on HF + LF (concatenated, unweighted)
    - MF_w : trained on HF + LF with variance-based scaling
    """
    # PDE-SINDy library in space
    X, T = np.meshgrid(x, t)
    XT = np.asarray([X, T]).T                # (Nt, Nx, 2)

    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )

    weak_lib = WeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        is_uniform=True,
        K=cfg.K,
        H_xt=cfg.H_xt,
        include_bias=cfg.include_bias,
    )

    def make_optimizer() -> ps.EnsembleOptimizer:
        return ps.EnsembleOptimizer(
            ps.STLSQ(threshold=cfg.stlsq_threshold),
            bagging=True,
            n_models=cfg.n_ensemble_models,
        )

    opt_hf   = make_optimizer()
    opt_lf   = make_optimizer()
    opt_mf   = make_optimizer()
    opt_mf_w = make_optimizer()

    # HF-only model
    model_hf = ps.SINDy(
        feature_library=weak_lib,
        optimizer=opt_hf,
    )
    model_hf.fit(X_hf, t=cfg.dt)

    # LF-only model
    model_lf = ps.SINDy(
        feature_library=weak_lib,
        optimizer=opt_lf,
    )
    model_lf.fit(X_lf, t=[t] * len(X_lf))

    # MF: HF + LF concatenated (unweighted)
    X_mf = list(X_hf) + list(X_lf)
    model_mf = ps.SINDy(
        feature_library=weak_lib,
        optimizer=opt_mf,
    )
    model_mf.fit(X_mf, t=[t] * len(X_mf))

    # MF_w: HF + LF with variance-based scaling
    eps_hf = max(noise_hf_abs, 1e-12)
    eps_lf = max(noise_lf_abs, 1e-12)

    w_hf = [(1 / eps_hf) ** 2 for _ in X_hf]
    w_lf = [(1 / eps_lf) ** 2 for _ in X_lf]
    w_mf = w_hf + w_lf

    model_mf_w = ps.SINDy(
        feature_library=weak_lib,
        optimizer=opt_mf_w,
    )
    model_mf_w.fit(X_mf, t=[t] * len(X_mf), sample_weight=w_mf)

    return (
        model_hf,
        model_lf,
        model_mf,
        model_mf_w,
        weak_lib,
        opt_hf,
        opt_lf,
        opt_mf,
        opt_mf_w,
    )


def _run_single_mf_run(
    run_idx: int,
    cfg: BurgersMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the Burgers multi-fidelity experiment.

    Returns
    -------
    errors : dict[str, (MAE, L0_error)]
        Keys: "HF", "LF", "MF", "MF_w".
    """
    # HF training set
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

    # LF training set (independent noise + IC seeds)
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

    # Ensemble PDE-SINDy models (HF-only, LF-only, MF, MF_w)
    (
        _model_hf,
        _model_lf,
        _model_mf,
        _model_mf_w,
        _pde_lib,
        opt_hf,
        opt_lf,
        opt_mf,
        opt_mf_w,
    ) = build_ensemble_sindy_models_burgers(
        X_hf=X_hf,
        X_lf=X_lf,
        t=t_train,
        x=x_grid,
        cfg=cfg,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
    )

    # Ensemble coefficients and medians
    coefs_hf   = np.array(opt_hf.coef_list)     # (n_models, n_targets, n_features)
    coefs_lf   = np.array(opt_lf.coef_list)
    coefs_mf   = np.array(opt_mf.coef_list)
    coefs_mf_w = np.array(opt_mf_w.coef_list)

    C_pred_hf   = np.median(coefs_hf,   axis=0)  # (n_targets, n_features)
    C_pred_lf   = np.median(coefs_lf,   axis=0)
    C_pred_mf   = np.median(coefs_mf,   axis=0)
    C_pred_mf_w = np.median(coefs_mf_w, axis=0)

    C_true = build_true_burgers_coefficients(cfg.nu)

    # MAE over all entries + L0 mismatch
    mae_hf,   l0_hf   = coefficient_errors(C_pred_hf,   C_true, relative_to_true_support=False)
    mae_lf,   l0_lf   = coefficient_errors(C_pred_lf,   C_true, relative_to_true_support=False)
    mae_mf,   l0_mf   = coefficient_errors(C_pred_mf,   C_true, relative_to_true_support=False)
    mae_mf_w, l0_mf_w = coefficient_errors(C_pred_mf_w, C_true, relative_to_true_support=False)

    errors = {
        "HF":   (mae_hf,   l0_hf),
        "LF":   (mae_lf,   l0_lf),
        "MF":   (mae_mf,   l0_mf),
        "MF_w": (mae_mf_w, l0_mf_w),
    }
    return errors


def run_burgers_mf_experiment(
    cfg: BurgersMFConfig,
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
    # Reference dataset for noise scaling
    X_ref_list, t_train, x_grid, _ = generate_burgers_dataset(
        n_traj=1,
        T=cfg.T_train,
        dt=cfg.dt,
        L=cfg.L,
        NX=cfg.NX,
        nu=cfg.nu,
        noise_level=0.0,
        seed=cfg.seed_base,
    )
    U_ref = X_ref_list[0]
    state_std = float(np.std(U_ref))

    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std

    models = ["HF", "LF", "MF", "MF_w"]

    def single_run(run_idx: int):
        return _run_single_mf_run(
            run_idx=run_idx,
            cfg=cfg,
            noise_hf_abs=noise_hf_abs,
            noise_lf_abs=noise_lf_abs,
        )

    df_errors, mae_errors, l0_errors = run_monte_carlo_experiment(
        n_runs=cfg.n_runs,
        methods=models,
        single_run_fn=single_run,
        results_dir=cfg.results_dir,
        results_filename=cfg.results_filename,
        metric1_name="MAE",
        metric2_name="L0",
        source_col="model",
        progress_desc="Monte Carlo Burgers MF",
    )

    return df_errors, mae_errors, l0_errors, state_std, noise_hf_abs, noise_lf_abs

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
    # RNG for this run
    if seed is None:
        rng = np.random.default_rng(cfg.seed_base)
    else:
        rng = np.random.default_rng(seed)

    # --- identical to _run_single_gls_run up to the model fits ----------
    x, t = make_space_time_grid(cfg)
    dx = x[1] - x[0]

    # random IC + clean trajectory
    u0 = random_initial_condition(rng, cfg)
    U_clean = burgers_solver(u0, cfg).T         # (NT, NX)
    U = U_clean[:, :, None]                     # (NT, NX, 1)

    # heteroscedastic noise ∝ |u_x|
    Ux = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2.0 * dx)
    grad_mag = np.abs(Ux[:, :, 0])              # (NT, NX)

    alpha = cfg.noise_level
    variance = (alpha * grad_mag) ** 2
    variance = np.maximum(variance, 1e-8)
    std = np.sqrt(variance)

    noise = std[:, :, None] * rng.standard_normal(size=U.shape)
    U_noisy = U + noise

    # spatiotemporal grid
    X, T = np.meshgrid(x, t)
    XT = np.asarray([X, T]).T                   # (NT, NX, 2)

    # base polynomial library in u
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )

    tf_seed = cfg.seed_base + 1000

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

    def make_optimizer() -> ps.EnsembleOptimizer:
        return ps.EnsembleOptimizer(
            ps.STLSQ(threshold=cfg.stlsq_threshold),
            bagging=True,
            n_models=cfg.n_ensemble_models,
        )

    opt_std  = make_optimizer()
    opt_var  = make_optimizer()
    opt_ones = make_optimizer()

    model_std  = ps.SINDy(feature_library=weak_lib,              optimizer=opt_std)
    model_var  = ps.SINDy(feature_library=weighted_weak_lib_var, optimizer=opt_var)
    model_ones = ps.SINDy(feature_library=weighted_weak_lib_ones,optimizer=opt_ones)

    model_std.fit(U_noisy, t=t)
    model_var.fit(U_noisy, t=t)
    model_ones.fit(U_noisy, t=t)

    # --- coefficients as 1D vectors (for PDE_rhs etc.) ------------------
    C_true = build_true_burgers_coefficients(cfg.nu).ravel()
    C_std  = np.array(model_std.optimizer.coef_,  copy=True).ravel()
    C_var  = np.array(model_var.optimizer.coef_,  copy=True).ravel()
    C_ones = np.array(model_ones.optimizer.coef_, copy=True).ravel()

    return C_true, C_std, C_var, C_ones
