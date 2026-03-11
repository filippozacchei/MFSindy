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

from mfsindy.cases.common import coefficient_errors, run_monte_carlo_experiment
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
class LorenzMFConfig:
    """Configuration for the Lorenz multi-fidelity SINDy experiment."""

    n_runs: int = 25

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
    results_dir: str = "results"
    results_filename: str = "lorenz_mf_errors.csv"


def build_ensemble_sindy_models_lorenz(
    X_hf: list[np.ndarray],
    X_lf: list[np.ndarray],
    t: np.ndarray,
    cfg: LorenzMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
):
    """
    Build four ensemble SINDy models for Lorenz:

    - HF   : HF trajectories only
    - LF   : LF trajectories only
    - MF   : HF + LF concatenated (unweighted)
    - MF_w : HF + LF with variance-based scaling
    """
    poly_lib = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )

    # Weak library on 1D time grid
    weak_lib = WeakPDELibrary(
        function_library=poly_lib,
        spatiotemporal_grid=t,
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

    model_hf = ps.SINDy(feature_library=weak_lib, optimizer=opt_hf)
    model_lf = ps.SINDy(feature_library=weak_lib, optimizer=opt_lf)
    model_mf = ps.SINDy(feature_library=weak_lib, optimizer=opt_mf)
    model_mf_w = ps.SINDy(feature_library=weak_lib, optimizer=opt_mf_w)

    # HF-only fit
    model_hf.fit(X_hf, t=cfg.dt)

    # LF-only fit
    model_lf.fit(X_lf, t=cfg.dt)

    # MF unweighted
    X_mf = list(X_hf) + list(X_lf)
    model_mf.fit(X_mf, t=cfg.dt)

    # MF weighted: inverse variance weights
    eps_hf = max(noise_hf_abs, 1e-12)
    eps_lf = max(noise_lf_abs, 1e-12)

    w_hf = (1.0 / eps_hf) ** 2
    w_lf = (1.0 / eps_lf) ** 2

    def _expand_weights(data_list: list[np.ndarray], weight: float) -> list[np.ndarray]:
        weights = []
        for traj in data_list:
            w = np.full(traj.shape[:-1], weight)
            if w.ndim == traj.ndim - 1:
                w = w[..., None]
            weights.append(w)
        return weights

    weights_hf = _expand_weights(X_hf, w_hf)
    weights_lf = _expand_weights(X_lf, w_lf)
    weights_mf = weights_hf + weights_lf

    model_mf_w.fit(X_mf, t=cfg.dt, sample_weight=weights_mf)

    return (
        model_hf,
        model_lf,
        model_mf,
        model_mf_w,
        poly_lib,
        opt_hf,
        opt_lf,
        opt_mf,
        opt_mf_w,
    )


def _run_single_lorenz_mf_run(
    run_idx: int,
    cfg: LorenzMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the Lorenz multi-fidelity experiment.
    """
    # HF training set
    X_hf, t_train, _ = generate_lorenz_dataset(
        n_traj=cfg.n_hf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_hf_abs,
        seed=cfg.seed_base + run_idx,
    )

    # LF training set (independent seeds)
    X_lf, _, _ = generate_lorenz_dataset(
        n_traj=cfg.n_lf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_lf_abs,
        seed=cfg.seed_base + 100 + run_idx,
    )

    (
        _model_hf,
        _model_lf,
        _model_mf,
        _model_mf_w,
        _poly_lib,
        opt_hf,
        opt_lf,
        opt_mf,
        opt_mf_w,
    ) = build_ensemble_sindy_models_lorenz(
        X_hf=X_hf,
        X_lf=X_lf,
        t=t_train,
        cfg=cfg,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
    )

    coefs_hf   = np.array(opt_hf.coef_list)   # (n_models, n_targets, n_features)
    coefs_lf   = np.array(opt_lf.coef_list)
    coefs_mf   = np.array(opt_mf.coef_list)
    coefs_mf_w = np.array(opt_mf_w.coef_list)

    coef_med_hf   = np.median(coefs_hf,   axis=0)   # (n_targets, n_features)
    coef_med_lf   = np.median(coefs_lf,   axis=0)
    coef_med_mf   = np.median(coefs_mf,   axis=0)
    coef_med_mf_w = np.median(coefs_mf_w, axis=0)

    # (n_features, n_targets)
    C_pred_hf   = coef_med_hf.T
    C_pred_lf   = coef_med_lf.T
    C_pred_mf   = coef_med_mf.T
    C_pred_mf_w = coef_med_mf_w.T

    C_true = build_true_coefficient_matrix()

    mae_hf,   l0_hf   = coefficient_errors(C_pred_hf,   C_true, relative_to_true_support=False)
    mae_lf,   l0_lf   = coefficient_errors(C_pred_lf,   C_true, relative_to_true_support=False)
    mae_mf,   l0_mf   = coefficient_errors(C_pred_mf,   C_true, relative_to_true_support=False)
    mae_mf_w, l0_mf_w = coefficient_errors(C_pred_mf_w, C_true, relative_to_true_support=False)

    return {
        "HF":   (mae_hf,   l0_hf),
        "LF":   (mae_lf,   l0_lf),
        "MF":   (mae_mf,   l0_mf),
        "MF_w": (mae_mf_w, l0_mf_w),
    }


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
    # Reference trajectory for noise scaling
    X_true_list, t_true, _ = generate_lorenz_dataset(
        n_traj=1,
        T=cfg.T_true,
        dt=cfg.dt,
        noise_level=0.0,
        seed=cfg.seed_base,
    )
    X_true = X_true_list[0]
    state_std = float(np.std(X_true))

    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std

    models = ["HF", "LF", "MF", "MF_w"]

    def single_run(run_idx: int):
        return _run_single_lorenz_mf_run(
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
        progress_desc="Monte Carlo Lorenz MF",
    )

    return df_errors, mae_errors, l0_errors, state_std, noise_hf_abs, noise_lf_abs

# ---------------------------------------------------------------------------
# Lorenz heteroscedastic GLS experiment (weak / weighted-weak SINDy)
# ---------------------------------------------------------------------------

@dataclass
class LorenzGLSConfig:
    """Configuration for the heteroscedastic Lorenz GLS experiment."""

    n_runs: int = 1000

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
    results_dir: str = "results"
    results_filename: str = "lorenz_weighted_errors.csv"


def _run_single_lorenz_gls_run(
    run_idx: int,
    cfg: LorenzGLSConfig,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the heteroscedastic Lorenz GLS experiment.

    Returns
    -------
    errors : dict[str, (L1_error, L0_error)]
        Keys: "No weighting", "Variance GLS", "Ones GLS".
    """
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

    # 4) Ensemble optimizers
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

    model_std.fit(U_noisy, t=t_eval)
    model_var.fit(U_noisy, t=t_eval)
    model_ones.fit(U_noisy, t=t_eval)
    
    model_var.print()

    # 5) True coefficients and errors
    C_true = build_true_coefficient_matrix()

    C_std = np.array(model_std.optimizer.coef_).T
    C_var = np.array(model_var.optimizer.coef_).T
    C_ones = np.array(model_ones.optimizer.coef_).T

    L1_std, L0_std = coefficient_errors(C_std, C_true, relative_to_true_support=True)
    L1_var, L0_var = coefficient_errors(C_var, C_true, relative_to_true_support=True)
    L1_ones, L0_ones = coefficient_errors(C_ones, C_true, relative_to_true_support=True)

    return {
        "No weighting": (L1_std, L0_std),
        "Variance GLS": (L1_var, L0_var),
        "Ones GLS": (L1_ones, L0_ones),
    }


def run_lorenz_gls_experiment(
    cfg: LorenzGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic Lorenz GLS experiment.

    Returns
    -------
    df_errors  : long-format DataFrame (run, method, metric, value)
    L1_errors  : dict[method] -> array of L1 errors
    L0_errors  : dict[method] -> array of L0 errors
    """
    methods = ["No weighting", "Variance GLS", "Ones GLS"]
    rng = np.random.default_rng(cfg.seed_base)

    def single_run(run_idx: int):
        return _run_single_lorenz_gls_run(run_idx=run_idx, cfg=cfg, rng=rng)

    df_errors, L1_errors, L0_errors = run_monte_carlo_experiment(
        n_runs=cfg.n_runs,
        methods=methods,
        single_run_fn=single_run,
        results_dir=cfg.results_dir,
        results_filename=cfg.results_filename,
        metric1_name="L1",
        metric2_name="L0",
        source_col="method",
        progress_desc="Monte Carlo Lorenz GLS",
    )

    return df_errors, L1_errors, L0_errors
