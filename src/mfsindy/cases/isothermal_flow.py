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
from scipy.integrate import solve_ivp

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
        H_xt=[L / 10.0, L / 10.0, T / 10.0],
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
class NSIsothermalMultiTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """
    Configuration for the isothermal NS multi-fidelity SINDy experiment
    (Part 1: HF / LF / MF / MF_w).
    """

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
    results_filename: str = "ns_isothermal_mf_errors.csv"

    def stlsq_kwargs(self) -> Dict[str, Any]:
        return {"alpha": self.stlsq_alpha}


def _ns_dataset_batch(
    run_idx: int,
    cfg: NSIsothermalMultiTrajectoryGLSConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
    *,
    grid: np.ndarray,
    t: np.ndarray,
) -> MultiTrajectoryGLSData:
    rng = np.random.default_rng(cfg.seed_base + 100 * run_idx)

    def _sample(n_traj: int, noise_abs: float, offset: int) -> list[np.ndarray]:
        data: list[np.ndarray] = []
        for j in range(n_traj):
            U_clean, _, _ = generate_isothermal_ns_dataset(
                N=cfg.N,
                Nt=cfg.Nt,
                L=cfg.L,
                T=cfg.T,
                mu=cfg.mu,
                RT=cfg.RT,
                seed=cfg.seed_base + run_idx * 1000 + offset + j,
                ic_type="taylor-green",
            )
            noise = noise_abs * rng.standard_normal(size=U_clean.shape)
            data.append(U_clean + noise)
        return data

    hf_data = _sample(cfg.n_hf, noise_hf_abs, offset=0)
    lf_data = _sample(cfg.n_lf, noise_lf_abs, offset=10_000)

    return MultiTrajectoryGLSData(
        hf=hf_data,
        lf=lf_data,
        t_argument=t,
        metadata={"grid": grid},
    )


def _ns_library(batch: MultiTrajectoryGLSData, cfg: NSIsothermalMultiTrajectoryGLSConfig):
    return WeakPDELibrary(
        function_library=_build_custom_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=batch.metadata["grid"],
        K=cfg.K,
        include_bias=cfg.include_bias,
    )


def run_ns_isothermal_multi_trajectory_gls_experiment(
    cfg: NSIsothermalMultiTrajectoryGLSConfig,
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
    C_true, _, _, _, _ = compute_reference_coefficients(
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
    def _reference_state_std(_: NSIsothermalMultiTrajectoryGLSConfig) -> float:
        return state_std

    def _dataset_builder(run_idx: int, cfg: NSIsothermalMultiTrajectoryGLSConfig, noise_hf: float, noise_lf: float):
        return _ns_dataset_batch(
            run_idx,
            cfg,
            noise_hf,
            noise_lf,
            grid=grid_ref,
            t=t_ref,
        )
    print(C_true)
    return run_multi_trajectory_gls_experiment(
        cfg,
        reference_state_std=_reference_state_std,
        dataset_builder=_dataset_builder,
        library_builder=_ns_library,
        true_coefficients=lambda _batch, _cfg: C_true,
        optimizer_factory=cfg.make_optimizer,
        progress_desc="MC isothermal NS MF",
    )


# ---------------------------------------------------------------------------
# PART 2: Heteroscedastic GLS experiment (as before)
# ---------------------------------------------------------------------------

@dataclass
class NSIsothermalIntraTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """
    Configuration for heteroscedastic GLS experiment on
    isothermal compressible Navier–Stokes (Part 2).
    """

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

    results_filename: str = "ns_isothermal_weighted_errors.csv"


def _build_ns_gls_artifacts(
    run_idx: int,
    cfg: NSIsothermalIntraTrajectoryGLSConfig,
    rng: np.random.Generator,
    *,
    grid: np.ndarray,
    base_library: ps.CustomLibrary,
    true_coefficients: np.ndarray,
) -> IntraTrajectoryGLSData:
    """Construct noisy NS data + weak libraries for one GLS run."""

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

    U_noisy, variance = add_heteroscedastic_noise_temporal_derivative(
        U_clean, t, sigma0=cfg.sigma0, alpha=cfg.alpha, rng=rng
    )

    variance_scaled = variance / np.mean(variance)
    tf_seed = cfg.seed_base + 1000 + run_idx
    h_xt = [cfg.L / 10.0, cfg.L / 10.0, cfg.T / 10.0]

    np.random.seed(tf_seed)
    weak_lib = WeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=h_xt,
        include_bias=cfg.include_bias,
    )

    np.random.seed(tf_seed)
    weighted_weak_lib_var = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        spatiotemporal_weights=variance_scaled,
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=h_xt,
        include_bias=cfg.include_bias,
    )

    np.random.seed(tf_seed)
    weighted_weak_lib_ones = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        spatiotemporal_weights=np.ones_like(variance_scaled),
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=h_xt,
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
        true_coefficients=true_coefficients,
    )


def run_ns_isothermal_intra_trajectory_gls_experiment(
    cfg: NSIsothermalIntraTrajectoryGLSConfig,
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

    def builder(run_idx: int, cfg: NSIsothermalIntraTrajectoryGLSConfig) -> IntraTrajectoryGLSData:
        rng = np.random.default_rng(cfg.seed_base + 100 * run_idx)
        return _build_ns_gls_artifacts(
            run_idx,
            cfg,
            rng,
            grid=grid_ref,
            base_library=base_library,
            true_coefficients=C_true,
        )

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="MC isothermal NS GLS",
    )
