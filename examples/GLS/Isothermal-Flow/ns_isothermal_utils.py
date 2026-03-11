# %% ns_isothermal_utils.py
"""
Isothermal compressible Navier–Stokes utilities.

Includes:

1) PDE + dataset generator
2) Heteroscedastic noise model based on temporal derivatives
3) Coefficient error metrics + generic MC wrapper
4) Multi-fidelity SINDy experiment (Part 1: HF / LF / MF / MF_w)
5) Heteroscedastic GLS experiment (Part 2: No weighting / Variance GLS / Ones GLS)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import solve_ivp

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary, WeightedWeakPDELibrary


# ---------------------------------------------------------------------------
# Core PDE: isothermal compressible Navier–Stokes
# ---------------------------------------------------------------------------

def compressible(t, U, dx, N, mu, RT):
    """
    2D isothermal compressible flow:

        state = (u, v, rho) on an N x N periodic grid.

    Parameters
    ----------
    U : (3 * N * N,) flattened state
        reshaped to (N, N, 3) as (u, v, rho).

    Returns
    -------
    dUdt_flat : flattened (N, N, 3) time derivative.
    """
    uvr = U.reshape(N, N, 3)
    u = uvr[:, :, 0]
    v = uvr[:, :, 1]
    rho = uvr[:, :, 2]

    FD1x = ps.differentiation.FiniteDifference(d=1, axis=0, periodic=True)
    FD1y = ps.differentiation.FiniteDifference(d=1, axis=1, periodic=True)
    FD2x = ps.differentiation.FiniteDifference(d=2, axis=0, periodic=True)
    FD2y = ps.differentiation.FiniteDifference(d=2, axis=1, periodic=True)

    ux  = FD1x._differentiate(u, dx)
    uy  = FD1y._differentiate(u, dx)
    uxx = FD2x._differentiate(u, dx)
    uyy = FD2y._differentiate(u, dx)

    vx  = FD1x._differentiate(v, dx)
    vy  = FD1y._differentiate(v, dx)
    vxx = FD2x._differentiate(v, dx)
    vyy = FD2y._differentiate(v, dx)

    p   = rho * RT
    px  = FD1x._differentiate(p, dx)
    py  = FD1y._differentiate(p, dx)

    ret = np.zeros_like(uvr)
    # u_t
    ret[:, :, 0] = -(u * ux + v * uy) - (px - mu * (uxx + uyy)) / rho
    # v_t
    ret[:, :, 1] = -(u * vx + v * vy) - (py - mu * (vxx + vyy)) / rho
    # rho_t
    ret[:, :, 2] = -(u * px / RT + v * py / RT + rho * ux + rho * vy)

    return ret.reshape(-1)


def make_initial_condition(
    X,
    Y,
    L,
    ic_type: str = "taylor-green",
    rng: np.random.Generator | None = None,
):
    """
    Return (U0, V0, RHO0) for a chosen flow configuration.

    X, Y : meshgrid on [0, L] x [0, L].
    """
    if rng is None:
        rng = np.random.default_rng()

    if ic_type == "taylor-green":
        U0 = (-np.sin(2 * np.pi / L * X) + 0.5 * np.cos(2 * 2 * np.pi / L * Y))
        V0 = (0.5 * np.cos(2 * np.pi / L * X) - np.sin(2 * 2 * np.pi / L * Y))
        RHO0 = 1.0 + 0.5 * np.cos(2 * np.pi / L * X) * np.cos(2 * 2 * np.pi / L * Y)

    elif ic_type == "shear-layer":
        U0 = np.tanh((Y - L / 2) / 0.1)
        V0 = 0.05 * np.sin(2 * np.pi * X / L)
        RHO0 = 1.0 + 0.1 * np.exp(-((Y - L / 2) ** 2) / (0.1 ** 2))

    else:
        raise ValueError(f"Unknown initial condition: {ic_type}")

    return U0, V0, RHO0


def generate_isothermal_ns_dataset(
    N: int = 64,
    Nt: int = 500,
    L: float = 5.0,
    T: float = 2.5,
    mu: float = 1.0,
    RT: float = 1.0,
    seed: int = 1,
    ic_type: str = "taylor-green",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single trajectory of isothermal compressible flow.

    Returns
    -------
    U_clean : (N, N, Nt, 3)
        States (u, v, rho).
    t : (Nt,)
        Time vector.
    grid : (N, N, Nt, 3)
        Spatiotemporal grid (x, y, t).
    """
    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, T, Nt)
    x = np.linspace(0.0, L, N, endpoint=False)
    y = np.linspace(0.0, L, N, endpoint=False)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Initial condition
    U0, V0, RHO0 = make_initial_condition(
        X, Y, L, ic_type=ic_type, rng=rng
    )
    y0 = np.zeros((N, N, 3))
    y0[:, :, 0] = U0
    y0[:, :, 1] = V0
    y0[:, :, 2] = RHO0

    sol = solve_ivp(
        compressible,
        (t[0], t[-1]),
        y0.reshape(-1),
        t_eval=t,
        args=(dx, N, mu, RT),
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )

    u_field = sol.y.reshape(N, N, 3, -1).transpose(0, 1, 3, 2)  # (N, N, Nt, 3)

    # Spatiotemporal grid
    grid = np.zeros((N, N, Nt, 3))
    grid[:, :, :, 0] = X[:, :, None]
    grid[:, :, :, 1] = Y[:, :, None]
    grid[:, :, :, 2] = t[None, None, :]

    return u_field, t, grid


# ---------------------------------------------------------------------------
# Heteroscedastic noise (Part 2): your derivative-based construction
# ---------------------------------------------------------------------------

def _ddt_centered(f: np.ndarray, dt: float) -> np.ndarray:
    """Centered finite difference in time with periodic wrap (axis=2)."""
    return (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * dt)


def add_heteroscedastic_noise_temporal_derivative(
    U_clean: np.ndarray,
    t: np.ndarray,
    sigma0: float = 1e-3,
    alpha: float = 0.025,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Noise model used in PART 2:

        time_deriv_mag = sqrt(u_t^2 + v_t^2)
        variance       = (sigma0 + alpha * time_deriv_mag)^2

    Noise is added to all state components (u, v, rho) with this variance.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = t[1] - t[0]

    u = U_clean[..., 0]
    v = U_clean[..., 1]

    u_t = _ddt_centered(u, dt)
    v_t = _ddt_centered(v, dt)

    time_deriv_mag = np.sqrt(u_t ** 2 + v_t ** 2)  # (N, N, Nt)

    variance = (sigma0 + alpha * time_deriv_mag) ** 2
    variance = np.maximum(variance, 1e-16)
    std = np.sqrt(variance)

    noise = std[..., None] * rng.standard_normal(size=U_clean.shape)
    U_noisy = U_clean + noise

    return U_noisy, variance


# ---------------------------------------------------------------------------
# Coefficient errors + generic MC wrapper
# ---------------------------------------------------------------------------

def coefficient_errors(
    C_est: np.ndarray,
    C_true: np.ndarray,
    tol_support: float = 1e-6,
    relative_to_true_support: bool = True,
) -> tuple[float, float]:
    """
    Mean absolute error on coefficients + L0 (support) mismatch.

    Shapes of C_est, C_true must match.
    """
    C_est = np.asarray(C_est)
    C_true = np.asarray(C_true)

    if C_est.shape != C_true.shape:
        raise ValueError(
            f"Shape mismatch in coefficient_errors: "
            f"C_est {C_est.shape}, C_true {C_true.shape}"
        )

    supp_true = np.abs(C_true) > tol_support
    supp_est = np.abs(C_est) > tol_support
    l0_mismatch = np.not_equal(supp_true, supp_est).astype(float)
    l0_err = float(np.mean(l0_mismatch))

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
    Generic Monte Carlo loop (same pattern as Hopf/Pendulum).
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
# Custom library + reference coefficients (used by both Part 1 and Part 2)
# ---------------------------------------------------------------------------

def _build_custom_library():
    """
    Custom feature library as in your code:

        library_functions = [x, 1 / (1e-6 + |x|)]
    """
    library_functions = [
        lambda x: x,
        lambda x: 1.0 / (1e-6 + np.abs(x)),
    ]
    library_function_names = [
        lambda x: x,
        lambda x: x + "^-1",
    ]
    base_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
    )
    return base_library


def compute_reference_coefficients(
    N: int,
    Nt: int,
    L: float,
    T: float,
    mu: float,
    RT: float,
    seed_base: int,
    derivative_order: int,
    include_bias: bool,
    p: int,
    K_ref: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ps.CustomLibrary]:
    """
    Clean reference model on one trajectory (weak SINDy).

    The resulting coefficients are used as "ground truth".
    """
    U_clean, t, grid = generate_isothermal_ns_dataset(
        N=N,
        Nt=Nt,
        L=L,
        T=T,
        mu=mu,
        RT=RT,
        seed=seed_base,
        ic_type="taylor-green",
    )

    base_library = _build_custom_library()

    weak_lib_ref = WeakPDELibrary(
        function_library=_build_custom_library(),  # separate instance is fine
        derivative_order=derivative_order,
        spatiotemporal_grid=grid,
        K=K_ref,
        p=p,
        # H_xt=[L / 10.0, L / 10.0, T / 10.0],
        include_bias=include_bias,
    )

    opt_ref = ps.STLSQ(threshold=0.5, alpha=1e-12)
    model_ref = ps.SINDy(feature_library=weak_lib_ref, optimizer=opt_ref)

    model_ref.fit(U_clean, t=t)
    model_ref.print()

    C_true = model_ref.optimizer.coef_.copy()  # shape (n_states=3, n_terms)
    return C_true, U_clean, t, grid, base_library


# ---------------------------------------------------------------------------
# PART 1: Multi-fidelity SINDy experiment (HF / LF / MF / MF_w)
# ---------------------------------------------------------------------------

@dataclass
class NSIsothermalMFConfig:
    """
    Configuration for the isothermal NS multi-fidelity SINDy experiment
    (Part 1: HF / LF / MF / MF_w).
    """

    n_runs: int = 10

    # multi-fidelity settings
    n_lf: int = 4
    n_hf: int = 1

    # relative noise levels w.r.t. std(U_clean_ref)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # grid / time
    N: int = 64
    Nt: int = 200
    Nt_std: int = 500
    L: float = 5.0
    T: float = 2.5
    T_std : float = 2.5

    # physical parameters
    mu: float = 1.0
    RT: float = 1.0

    # weak-library settings
    derivative_order: int = 2
    include_bias: bool = False
    p: int = 2
    K: int = 500
    K_std: int = 500

    # SINDy / optimizer settings
    stlsq_threshold: float = 0.5
    stlsq_alpha: float = 1e-8
    n_ensemble_models: int = 50

    # randomness
    seed_base: int = 1

    # output
    results_dir: str = "results"
    results_filename: str = "ns_isothermal_mf_errors.csv"


def _build_ensemble_models_ns(
    X_hf: List[np.ndarray],
    X_lf: List[np.ndarray],
    grid: np.ndarray,
    t: np.ndarray,
    cfg: NSIsothermalMFConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
):
    """
    Build four ensemble SINDy models for NS:

        - HF   : HF trajectories only
        - LF   : LF trajectories only
        - MF   : HF + LF concatenated (unweighted)
        - MF_w : HF + LF with variance-based scaling
    """
    base_library = _build_custom_library()

    weak_lib = WeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        K=cfg.K,
        # H_xt=[cfg.L / 10.0, cfg.L / 10.0, cfg.T / 10.0],
        include_bias=cfg.include_bias,
    )

    def make_optimizer() -> ps.EnsembleOptimizer:
        return ps.EnsembleOptimizer(
            ps.STLSQ(threshold=cfg.stlsq_threshold, alpha=cfg.stlsq_alpha),
            bagging=True,
            n_models=cfg.n_ensemble_models,
        )

    opt_hf   = make_optimizer()
    opt_lf   = make_optimizer()
    opt_mf   = make_optimizer()
    opt_mf_w = make_optimizer()

    model_hf   = ps.SINDy(feature_library=weak_lib, optimizer=opt_hf)
    model_lf   = ps.SINDy(feature_library=weak_lib, optimizer=opt_lf)
    model_mf   = ps.SINDy(feature_library=weak_lib, optimizer=opt_mf)
    model_mf_w = ps.SINDy(feature_library=weak_lib, optimizer=opt_mf_w)

    # HF-only fit
    model_hf.fit(X_hf, t=t)
    model_hf.print()

    # LF-only fit
    model_lf.fit(X_lf, t=t)
    model_lf.print()

    # MF unweighted
    X_mf = list(X_hf) + list(X_lf)
    model_mf.fit(X_mf, t=t)
    model_mf.print()

    # MF weighted by inverse variance (homoscedastic noise per fidelity)
    eps_hf = max(noise_hf_abs, 1e-12)
    eps_lf = max(noise_lf_abs, 1e-12)

    w_hf = (1.0 / eps_hf) ** 2
    print(w_hf)
    w_lf = (1.0 / eps_lf) ** 2
    print(w_lf)
    

    weights_hf = [w_hf for _ in X_hf]
    weights_lf = [w_lf for _ in X_lf]
    weights_mf = weights_hf + weights_lf

    model_mf_w.fit(X_mf, t=t, sample_weight=weights_mf)
    model_mf_w.print()

    return (
        model_hf,
        model_lf,
        model_mf,
        model_mf_w,
        opt_hf,
        opt_lf,
        opt_mf,
        opt_mf_w,
    )


def _run_single_ns_mf_run(
    run_idx: int,
    cfg: NSIsothermalMFConfig,
    C_true: np.ndarray,
    grid: np.ndarray,
    t: np.ndarray,
    state_std: float,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the NS multi-fidelity experiment.

    HF / LF differ only in homoscedastic noise level.
    """
    rng = np.random.default_rng(cfg.seed_base + 100 * run_idx)

    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std

    # HF trajectories
    X_hf: List[np.ndarray] = []
    for j in range(cfg.n_hf):
        U_clean, _, _ = generate_isothermal_ns_dataset(
            N=cfg.N,
            Nt=cfg.Nt,
            L=cfg.L,
            T=cfg.T,
            mu=cfg.mu,
            RT=cfg.RT,
            seed=cfg.seed_base + run_idx * 1000 + j,
            ic_type="taylor-green",
        )
        U_noisy = U_clean + noise_hf_abs * rng.standard_normal(size=U_clean.shape)
        X_hf.append(U_noisy)
    print("HF Generated")
    

    # LF trajectories
    X_lf: List[np.ndarray] = []
    for j in range(cfg.n_lf):
        U_clean, _, _ = generate_isothermal_ns_dataset(
            N=cfg.N,
            Nt=cfg.Nt,
            L=cfg.L,
            T=cfg.T,
            mu=cfg.mu,
            RT=cfg.RT,
            seed=cfg.seed_base + run_idx * 1000 + 10000 + j,
            ic_type="taylor-green",
        )
        U_noisy = U_clean + noise_lf_abs * rng.standard_normal(size=U_clean.shape)
        X_lf.append(U_noisy)
    print("LF Generated")

    (
        model_hf,
        model_lf,
        model_mf,
        model_mf_w,
        opt_hf,
        opt_lf,
        opt_mf,
        opt_mf_w,
    ) = _build_ensemble_models_ns(
        X_hf=X_hf,
        X_lf=X_lf,
        grid=grid,
        t=t,
        cfg=cfg,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
    )

    # Ensemble median coefficients: (n_states, n_terms)
    coefs_hf   = opt_hf.coef_   
    coefs_lf   = opt_lf.coef_
    coefs_mf   = opt_mf.coef_
    coefs_mf_w = opt_mf_w.coef_

    mae_hf,   l0_hf   = coefficient_errors(coefs_hf,   C_true, relative_to_true_support=True)
    mae_lf,   l0_lf   = coefficient_errors(coefs_lf,   C_true, relative_to_true_support=True)
    mae_mf,   l0_mf   = coefficient_errors(coefs_mf,   C_true, relative_to_true_support=True)
    mae_mf_w, l0_mf_w = coefficient_errors(coefs_mf_w, C_true, relative_to_true_support=True)

    return {
        "HF":   (mae_hf,   l0_hf),
        "LF":   (mae_lf,   l0_lf),
        "MF":   (mae_mf,   l0_mf),
        "MF_w": (mae_mf_w, l0_mf_w),
    }


def run_ns_isothermal_mf_experiment(
    cfg: NSIsothermalMFConfig,
) -> tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    float,
    float,
    float,
]:
    """
    Full NS multi-fidelity experiment.

    Returns
    -------
    df_errors    : long-format DataFrame (run, model, metric, value)
    mae_errors   : dict[model] -> array of MAE errors
    l0_errors    : dict[model] -> array of L0 errors
    state_std    : reference state standard deviation
    noise_hf_abs : absolute HF noise level
    noise_lf_abs : absolute LF noise level
    """
    # Reference clean dataset for state_std and "truth"
    C_true, U_ref, t_ref, grid_ref, _ = compute_reference_coefficients(
        N=cfg.N,
        Nt=cfg.Nt_std,
        L=cfg.L,
        T=cfg.T_std,
        mu=cfg.mu,
        RT=cfg.RT,
        seed_base=cfg.seed_base,
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        p=cfg.p,
        K_ref=cfg.K_std,
    )
    
    _, U_ref, t_ref, grid_ref, _ = compute_reference_coefficients(
        N=cfg.N,
        Nt=cfg.Nt,
        L=cfg.L,
        T=cfg.T,
        mu=cfg.mu,
        RT=cfg.RT,
        seed_base=cfg.seed_base,
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        p=cfg.p,
        K_ref=cfg.K,
    )


    state_std = float(np.std(U_ref))
    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std

    models = ["HF", "LF", "MF", "MF_w"]

    def single_run(run_idx: int):
        return _run_single_ns_mf_run(
            run_idx=run_idx,
            cfg=cfg,
            C_true=C_true,
            grid=grid_ref,
            t=t_ref,
            state_std=state_std,
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
        progress_desc="MC isothermal NS MF",
    )

    return df_errors, mae_errors, l0_errors, state_std, noise_hf_abs, noise_lf_abs


# ---------------------------------------------------------------------------
# PART 2: Heteroscedastic GLS experiment (as before)
# ---------------------------------------------------------------------------

@dataclass
class NSIsothermalGLSConfig:
    """
    Configuration for heteroscedastic GLS experiment on
    isothermal compressible Navier–Stokes (Part 2).
    """

    n_runs: int = 20

    # grid / time
    N: int = 64
    Nt: int = 500
    L: float = 5.0
    T: float = 2.5

    # physical parameters
    mu: float = 1.0
    RT: float = 1.0

    # heteroscedastic noise parameters
    sigma0: float = 1e-3
    alpha: float = 0.025

    # weak-library settings
    derivative_order: int = 2
    include_bias: bool = False
    p: int = 2
    K_ref: int = 1000
    K: int = 1000

    # SINDy / optimizer settings
    stlsq_threshold: float = 0.5
    stlsq_alpha: float = 1e-8
    n_ensemble_models: int = 100

    # randomness
    seed_base: int = 1

    # output
    results_dir: str = "results"
    results_filename: str = "ns_isothermal_weighted_errors.csv"


def _run_single_ns_gls_run(
    run_idx: int,
    cfg: NSIsothermalGLSConfig,
    C_true: np.ndarray,
    grid: np.ndarray,
    base_library: ps.CustomLibrary,
) -> Dict[str, Tuple[float, float]]:
    """
    One Monte Carlo run of the heteroscedastic NS GLS experiment.

    Uses the derivative-based heteroscedastic noise model.
    """
    rng = np.random.default_rng(cfg.seed_base + 100 * run_idx)

    U_clean, t, _ = generate_isothermal_ns_dataset(
        N=cfg.N,
        Nt=cfg.Nt,
        L=cfg.L,
        T=cfg.T,
        mu=cfg.mu,
        RT=cfg.RT,
        seed=cfg.seed_base + run_idx + 1,
        ic_type="taylor-green",
    )
    print("HF Generated")

    U_noisy, variance = add_heteroscedastic_noise_temporal_derivative(
        U_clean, t, sigma0=cfg.sigma0, alpha=cfg.alpha, rng=rng
    )

    variance_scaled = variance / np.mean(variance)

    np.random.seed(cfg.seed_base + 1000 + run_idx)
    weak_lib = WeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=[cfg.L / 10.0, cfg.L / 10.0, cfg.T / 40.0],
        include_bias=cfg.include_bias,
    )

    np.random.seed(cfg.seed_base + 1000 + run_idx)
    weighted_weak_lib_var = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        spatiotemporal_weights=variance_scaled,
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=[cfg.L / 10.0, cfg.L / 10.0, cfg.T / 40.0],
        include_bias=cfg.include_bias,
    )

    np.random.seed(cfg.seed_base + 1000 + run_idx)
    weighted_weak_lib_ones = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        spatiotemporal_weights=np.ones_like(variance_scaled),
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=[cfg.L / 10.0, cfg.L / 10.0, cfg.T / 40.0],
        include_bias=cfg.include_bias,
    )

    def make_optimizer() -> ps.EnsembleOptimizer:
        return ps.EnsembleOptimizer(
            ps.STLSQ(threshold=cfg.stlsq_threshold, alpha=cfg.stlsq_alpha),
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
    model_std.print()
    model_var.fit(U_noisy, t=t)
    model_var.print()
    model_ones.fit(U_noisy, t=t)
    model_ones.print()

    C_std = np.array(model_std.optimizer.coef_)
    C_var = np.array(model_var.optimizer.coef_)
    C_ones = np.array(model_ones.optimizer.coef_)

    L1_std, L0_std = coefficient_errors(C_std, C_true, relative_to_true_support=True)
    L1_var, L0_var = coefficient_errors(C_var, C_true, relative_to_true_support=True)
    L1_ones, L0_ones = coefficient_errors(
        C_ones, C_true, relative_to_true_support=True
    )

    return {
        "No weighting": (L1_std, L0_std),
        "Variance GLS": (L1_var, L0_var),
        "Ones GLS": (L1_ones, L0_ones),
    }


def run_ns_isothermal_gls_experiment(
    cfg: NSIsothermalGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic NS GLS experiment (Part 2).
    """
    C_true, U_clean_ref, t_ref, grid_ref, base_library = compute_reference_coefficients(
        N=cfg.N,
        Nt=cfg.Nt,
        L=cfg.L,
        T=cfg.T,
        mu=cfg.mu,
        RT=cfg.RT,
        seed_base=cfg.seed_base,
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        p=cfg.p,
        K_ref=cfg.K_ref,
    )

    methods = ["No weighting", "Variance GLS", "Ones GLS"]

    def single_run(run_idx: int):
        return _run_single_ns_gls_run(
            run_idx=run_idx,
            cfg=cfg,
            C_true=C_true,
            grid=grid_ref,
            base_library=base_library,
        )

    df_errors, L1_errors, L0_errors = _run_monte_carlo_experiment(
        n_runs=cfg.n_runs,
        methods=methods,
        single_run_fn=single_run,
        results_dir=cfg.results_dir,
        results_filename=cfg.results_filename,
        metric1_name="L1",
        metric2_name="L0",
        source_col="method",
        progress_desc="MC isothermal NS GLS",
    )

    return df_errors, L1_errors, L0_errors
