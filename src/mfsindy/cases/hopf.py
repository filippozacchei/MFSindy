# ---------------------------------------------------------------------------
# Hopf oscillator: dynamics, trajectories, true coefficients
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary

from mfsindy.weighted_weak_pde_library import WeightedWeakPDELibrary

from scipy.integrate import solve_ivp  # at top of file if not already imported

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
# Hopf multi-fidelity experiment (HF / LF / MF / MF_w)
# ---------------------------------------------------------------------------
@dataclass
class HopfMFConfig:
    """Configuration for the Hopf multi-fidelity SINDy experiment."""

    n_runs: int = 25

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
    results_dir: str = "results"
    results_filename: str = "hopf_mf_errors.csv"

def build_ensemble_sindy_models_hopf(
    X_hf: list[np.ndarray],
    X_lf: list[np.ndarray],
    t: np.ndarray,
    cfg: HopfMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
):
    """
    Build four ensemble SINDy models for Hopf:

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

    # MF weighted by inverse variance
    eps_hf = max(noise_hf_abs, 1e-12)
    eps_lf = max(noise_lf_abs, 1e-12)

    w_hf = (1.0 / eps_hf) ** 2
    w_lf = (1.0 / eps_lf) ** 2

    weights_hf = [w_hf for _ in X_hf]
    weights_lf = [w_lf for _ in X_lf]
    weights_mf = weights_hf + weights_lf

    model_mf_w.fit(X_mf, t=cfg.dt, sample_weight=weights_mf)

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

def _run_single_hopf_mf_run(
    run_idx: int,
    cfg: HopfMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the Hopf multi-fidelity experiment.

    Returns
    -------
    errors : dict[str, (MAE, L0_error)]
        Keys: "HF", "LF", "MF", "MF_w".
    """
    # HF training set
    X_hf, t_train, _ = generate_hopf_dataset(
        n_traj=cfg.n_hf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_hf_abs,
        seed=cfg.seed_base + run_idx,
        mu=cfg.mu,
        omega=cfg.omega,
    )

    # LF training set (independent seeds)
    X_lf, _, _ = generate_hopf_dataset(
        n_traj=cfg.n_lf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=noise_lf_abs,
        seed=cfg.seed_base + 100 + run_idx,
        mu=cfg.mu,
        omega=cfg.omega,
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
    ) = build_ensemble_sindy_models_hopf(
        X_hf=X_hf,
        X_lf=X_lf,
        t=t_train,
        cfg=cfg,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
    )

    C_pred_hf = np.array(opt_hf.coef_).T
    C_pred_lf = np.array(opt_lf.coef_).T
    C_pred_mf = np.array(opt_mf.coef_).T
    C_pred_mf_w = np.array(opt_mf_w.coef_).T

    C_true = build_true_hopf_coefficients(mu=cfg.mu, omega=cfg.omega)

    mae_hf,   l0_hf   = coefficient_errors(C_pred_hf,   C_true, relative_to_true_support=True)
    mae_lf,   l0_lf   = coefficient_errors(C_pred_lf,   C_true, relative_to_true_support=True)
    mae_mf,   l0_mf   = coefficient_errors(C_pred_mf,   C_true, relative_to_true_support=True)
    mae_mf_w, l0_mf_w = coefficient_errors(C_pred_mf_w, C_true, relative_to_true_support=True)

    return {
        "HF":   (mae_hf,   l0_hf),
        "LF":   (mae_lf,   l0_lf),
        "MF":   (mae_mf,   l0_mf),
        "MF_w": (mae_mf_w, l0_mf_w),
    }


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
    # Reference trajectory for noise scaling
    X_ref_list, t_ref, _ = generate_hopf_dataset(
        n_traj=1,
        T=cfg.T_true,
        dt=cfg.dt,
        noise_level=0.0,
        seed=cfg.seed_base,
        mu=cfg.mu,
        omega=cfg.omega,
    )
    U_ref = X_ref_list[0]
    state_std = float(np.std(U_ref))

    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std

    models = ["HF", "LF", "MF", "MF_w"]

    def single_run(run_idx: int):
        return _run_single_hopf_mf_run(
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
        progress_desc="Monte Carlo Hopf MF",
    )

    return df_errors, mae_errors, l0_errors, state_std, noise_hf_abs, noise_lf_abs

# ---------------------------------------------------------------------------
# Hopf heteroscedastic GLS experiment (weak / weighted-weak SINDy)
# ---------------------------------------------------------------------------

@dataclass
class HopfGLSConfig:
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

    # random seeds
    seed_base: int = 0

    # output
    results_dir: str = "results"
    results_filename: str = "hopf_weighted_errors.csv"
    
def _run_single_hopf_gls_run(
    run_idx: int,
    cfg: HopfGLSConfig,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the heteroscedastic Hopf GLS experiment.

    Returns
    -------
    errors : dict[str, (L1_error, L0_error)]
        Keys: "No weighting", "Variance GLS", "Ones GLS".
    """
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

    C_true = build_true_hopf_coefficients(mu=cfg.mu, omega=cfg.omega)

    C_std = np.array(model_std.optimizer.coef_).T
    C_var = np.array(model_var.optimizer.coef_).T
    C_ones = np.array(model_ones.optimizer.coef_).T

    L1_std, L0_std = coefficient_errors(C_std,  C_true, relative_to_true_support=True)
    L1_var, L0_var = coefficient_errors(C_var,  C_true, relative_to_true_support=True)
    L1_ones, L0_ones = coefficient_errors(C_ones, C_true, relative_to_true_support=True)

    return {
        "No weighting": (L1_std, L0_std),
        "Variance GLS": (L1_var, L0_var),
        "Ones GLS": (L1_ones, L0_ones),
    }


def run_hopf_gls_experiment(
    cfg: HopfGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic Hopf GLS experiment.

    Returns
    -------
    df_errors  : long-format DataFrame (run, method, metric, value)
    L1_errors  : dict[method] -> array of L1 errors
    L0_errors  : dict[method] -> array of L0 errors
    """
    methods = ["No weighting", "Variance GLS", "Ones GLS"]
    rng = np.random.default_rng(cfg.seed_base)

    def single_run(run_idx: int):
        return _run_single_hopf_gls_run(run_idx=run_idx, cfg=cfg, rng=rng)

    df_errors, L1_errors, L0_errors = _run_monte_carlo_experiment(
        n_runs=cfg.n_runs,
        methods=methods,
        single_run_fn=single_run,
        results_dir=cfg.results_dir,
        results_filename=cfg.results_filename,
        metric1_name="L1",
        metric2_name="L0",
        source_col="method",
        progress_desc="Monte Carlo Hopf GLS",
    )

    return df_errors, L1_errors, L0_errors
