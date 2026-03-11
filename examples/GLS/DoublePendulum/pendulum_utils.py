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
from tqdm import tqdm

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary, WeightedWeakPDELibrary

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
# Generic coefficient error function + MC wrapper
# ---------------------------------------------------------------------------


def coefficient_errors(
    C_est: np.ndarray,
    C_true: np.ndarray,
    tol_support: float = 1e-6,
    relative_to_true_support: bool = False,
) -> tuple[float, float]:
    """
    Error on coefficients and L0 (support) mismatch.

    Parameters
    ----------
    C_est, C_true : arrays of same shape
    tol_support : float
        Threshold for deciding nonzero support.
    relative_to_true_support : bool
        If False: MAE over all entries.
        If True : MAE over entries where |C_true| > tol_support
                  (L1 on true support).

    Returns
    -------
    err : float
        Mean absolute error (as defined above).
    l0_err : float
        Mean support mismatch (zero vs non-zero pattern).
    """
    C_est = np.asarray(C_est)
    C_true = np.asarray(C_true)

    if C_est.shape != C_true.shape:
        raise ValueError(
            f"Shape mismatch in coefficient_errors: "
            f"C_est {C_est.shape}, C_true {C_true.shape}"
        )

    # L0 support mismatch
    supp_true = np.abs(C_true) > tol_support
    supp_est = np.abs(C_est) > tol_support
    l0_mismatch = np.not_equal(supp_true, supp_est).astype(float)
    l0_err = float(np.mean(l0_mismatch))

    # Error value
    if relative_to_true_support:
        if np.any(supp_true):
            C_true_nz = C_true[supp_true]
            C_est_nz = C_est[supp_true]
            err = float(np.mean(np.abs(C_est_nz - C_true_nz)))
        else:
            err = 0.0
    else:
        err = float(np.mean(np.abs(C_est - C_true)))

    return err, l0_err


def _run_monte_carlo_experiment(
    n_runs: int,
    methods: List[str],
    single_run_fn: Callable[[int], Dict[str, Tuple[float, float]]],
    *,
    results_dir: str,
    results_filename: str,
    metric1_name: str,
    metric2_name: str,
    source_col: str,
    progress_desc: str,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Generic Monte Carlo loop:

    - calls `single_run_fn(run_idx)` for run_idx = 0, ..., n_runs - 1,
      where each call returns a dict[method] -> (metric1, metric2)
    - accumulates errors into dicts of arrays
    - saves long-format CSV with schema (run, source_col, metric, value)
    - returns DataFrame and error dicts for plotting.
    """
    metric1_errors: Dict[str, List[float]] = {m: [] for m in methods}
    metric2_errors: Dict[str, List[float]] = {m: [] for m in methods}

    for k in tqdm(range(n_runs), desc=progress_desc):
        errs = single_run_fn(k)
        for m in methods:
            e1, e2 = errs[m]
            metric1_errors[m].append(e1)
            metric2_errors[m].append(e2)

    os.makedirs(results_dir, exist_ok=True)
    errors_path = os.path.join(results_dir, results_filename)

    rows = []
    for m in methods:
        m1_arr = np.asarray(metric1_errors[m])
        m2_arr = np.asarray(metric2_errors[m])
        for run_id, (e1, e2) in enumerate(zip(m1_arr, m2_arr)):
            rows.append(
                {"run": run_id, source_col: m, "metric": metric1_name, "value": e1}
            )
            rows.append(
                {"run": run_id, source_col: m, "metric": metric2_name, "value": e2}
            )

    df_errors = pd.DataFrame(rows)
    df_errors.to_csv(errors_path, index=False)

    metric1_arrs = {m: np.asarray(metric1_errors[m]) for m in methods}
    metric2_arrs = {m: np.asarray(metric2_errors[m]) for m in methods}

    return df_errors, metric1_arrs, metric2_arrs


# ---------------------------------------------------------------------------
# Pendulum multi-fidelity experiment (HF / LF / MF / MF_w)
# ---------------------------------------------------------------------------


@dataclass
class PendulumMFConfig:
    """Configuration for the pendulum multi-fidelity SINDy experiment."""

    n_runs: int = 25

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
    results_dir: str = "results"
    results_filename: str = "pendulum_mf_errors.csv"


def build_ensemble_sindy_models_pendulum(
    X_hf: list[np.ndarray],
    X_lf: list[np.ndarray],
    t: np.ndarray,
    cfg: PendulumMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
):
    """
    Build four ensemble SINDy models for the pendulum:

    - HF   : HF trajectories only
    - LF   : LF trajectories only
    - MF   : HF + LF concatenated (unweighted)
    - MF_w : HF + LF with variance-based scaling
    """
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )

    weak_lib = WeakPDELibrary(
        function_library=base_library,
        spatiotemporal_grid=t,
    )

    def make_optimizer() -> ps.EnsembleOptimizer:
        return ps.EnsembleOptimizer(
            ps.STLSQ(threshold=cfg.stlsq_threshold),
            bagging=True,
            n_models=cfg.n_ensemble_models,
        )

    opt_hf = make_optimizer()
    opt_lf = make_optimizer()
    opt_mf = make_optimizer()
    opt_mf_w = make_optimizer()

    model_hf = ps.SINDy(feature_library=weak_lib, optimizer=opt_hf)
    model_lf = ps.SINDy(feature_library=weak_lib, optimizer=opt_lf)
    model_mf = ps.SINDy(feature_library=weak_lib, optimizer=opt_mf)
    model_mf_w = ps.SINDy(feature_library=weak_lib, optimizer=opt_mf_w)

    # HF-only fit
    model_hf.fit(X_hf, t=cfg.dt)

    # LF-only fit
    model_lf.fit(X_lf, t=cfg.dt)

    # MF unweighted (concatenate)
    X_mf = list(X_hf) + list(X_lf)
    model_mf.fit(X_mf, t=cfg.dt)

    # MF weighted by inverse variance (per-trajectory weights)
    eps_hf = max(noise_hf_abs, 1e-12)
    eps_lf = max(noise_lf_abs, 1e-12)
    w_hf = (1.0 / eps_hf) ** 2
    w_lf = (1.0 / eps_lf) ** 2

    weights_hf = [w_hf for _ in X_hf]
    weights_lf = [w_lf for _ in X_lf]
    weights_mf = weights_hf + weights_lf

    model_mf_w.fit(X_mf, t=cfg.dt, sample_weight=weights_mf)
    
    model_hf.print()
    model_lf.print()
    model_mf.print()
    model_mf_w.print()

    return (
        model_hf,
        model_lf,
        model_mf,
        model_mf_w,
        base_library,
        opt_hf,
        opt_lf,
        opt_mf,
        opt_mf_w,
    )


def _run_single_pendulum_mf_run(
    run_idx: int,
    cfg: PendulumMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the pendulum multi-fidelity experiment.

    Returns
    -------
    errors : dict[str, (MAE, L0_error)]
        Keys: "HF", "LF", "MF", "MF_w".
    """
    # HF training set
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

    # LF training set (independent seeds)
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
    ) = build_ensemble_sindy_models_pendulum(
        X_hf=X_hf,
        X_lf=X_lf,
        t=t_train,
        cfg=cfg,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
    )

    # EnsembleOptimizer.coef_ is the ensemble-aggregated coefficient (e.g. median)
    C_pred_hf = np.array(opt_hf.coef_).T
    C_pred_lf = np.array(opt_lf.coef_).T
    C_pred_mf = np.array(opt_mf.coef_).T
    C_pred_mf_w = np.array(opt_mf_w.coef_).T

    C_true = build_true_pendulum_coefficients(g=cfg.g, L=cfg.L, c=cfg.c)

    mae_hf, l0_hf = coefficient_errors(
        C_pred_hf, C_true, relative_to_true_support=True
    )
    mae_lf, l0_lf = coefficient_errors(
        C_pred_lf, C_true, relative_to_true_support=True
    )
    mae_mf, l0_mf = coefficient_errors(
        C_pred_mf, C_true, relative_to_true_support=True
    )
    mae_mf_w, l0_mf_w = coefficient_errors(
        C_pred_mf_w, C_true, relative_to_true_support=True
    )

    return {
        "HF": (mae_hf, l0_hf),
        "LF": (mae_lf, l0_lf),
        "MF": (mae_mf, l0_mf),
        "MF_w": (mae_mf_w, l0_mf_w),
    }


def run_pendulum_mf_experiment(
    cfg: PendulumMFConfig,
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
    # Reference trajectory for noise scaling
    X_ref_list, t_ref, _ = generate_pendulum_dataset(
        n_traj=1,
        T=cfg.T_true,
        dt=cfg.dt,
        noise_level=0.0,
        seed=cfg.seed_base,
        g=cfg.g,
        L=cfg.L,
        c=cfg.c,
    )
    U_ref = X_ref_list[0]
    state_std = float(np.std(U_ref))

    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std

    models = ["HF", "LF", "MF", "MF_w"]

    def single_run(run_idx: int):
        return _run_single_pendulum_mf_run(
            run_idx=run_idx,
            cfg=cfg,
            noise_hf_abs=noise_hf_abs,
            noise_lf_abs=noise_lf_abs,
        )

    df_errors, mae_errors, l0_errors = _run_monte_carlo_experiment(
        n_runs=cfg.n_runs,
        methods=models,
        single_run_fn=single_run,
        results_dir=cfg.results_dir,
        results_filename=cfg.results_filename,
        metric1_name="MAE",
        metric2_name="L0",
        source_col="model",
        progress_desc="Monte Carlo pendulum MF",
    )

    return df_errors, mae_errors, l0_errors, state_std, noise_hf_abs, noise_lf_abs


# ---------------------------------------------------------------------------
# Pendulum heteroscedastic GLS experiment (weak / weighted-weak SINDy)
# ---------------------------------------------------------------------------


@dataclass
class PendulumGLSConfig:
    """Configuration for the heteroscedastic pendulum GLS experiment."""

    n_runs: int = 100

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

    # random seeds
    seed_base: int = 0

    # output
    results_dir: str = "results"
    results_filename: str = "pendulum_weighted_errors.csv"


def _run_single_pendulum_gls_run(
    run_idx: int,
    cfg: PendulumGLSConfig,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the heteroscedastic pendulum GLS experiment.

    Returns
    -------
    errors : dict[str, (L1_error, L0_error)]
        Keys: "No weighting", "Variance GLS", "Ones GLS".
    """
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

    model_std.fit(Y_noisy, t=t_eval)
    model_var.fit(Y_noisy, t=t_eval)
    model_ones.fit(Y_noisy, t=t_eval)

    C_true = build_true_pendulum_coefficients(g=cfg.g, L=cfg.L, c=cfg.c)

    C_std = np.array(model_std.optimizer.coef_).T
    C_var = np.array(model_var.optimizer.coef_).T
    C_ones = np.array(model_ones.optimizer.coef_).T

    L1_std, L0_std = coefficient_errors(
        C_std, C_true, relative_to_true_support=True
    )
    L1_var, L0_var = coefficient_errors(
        C_var, C_true, relative_to_true_support=True
    )
    L1_ones, L0_ones = coefficient_errors(
        C_ones, C_true, relative_to_true_support=True
    )

    return {
        "No weighting": (L1_std, L0_std),
        "Variance GLS": (L1_var, L0_var),
        "Ones GLS": (L1_ones, L0_ones),
    }


def run_pendulum_gls_experiment(
    cfg: PendulumGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic pendulum GLS experiment.

    Returns
    -------
    df_errors  : long-format DataFrame (run, method, metric, value)
    L1_errors  : dict[method] -> array of L1 errors
    L0_errors  : dict[method] -> array of L0 errors
    """
    methods = ["No weighting", "Variance GLS", "Ones GLS"]
    rng = np.random.default_rng(cfg.seed_base)

    def single_run(run_idx: int):
        return _run_single_pendulum_gls_run(run_idx=run_idx, cfg=cfg, rng=rng)

    df_errors, L1_errors, L0_errors = _run_monte_carlo_experiment(
        n_runs=cfg.n_runs,
        methods=methods,
        single_run_fn=single_run,
        results_dir=cfg.results_dir,
        results_filename=cfg.results_filename,
        metric1_name="L1",
        metric2_name="L0",
        source_col="method",
        progress_desc="Monte Carlo pendulum GLS",
    )

    return df_errors, L1_errors, L0_errors
