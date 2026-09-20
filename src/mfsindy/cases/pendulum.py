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
from dataclasses import dataclass, replace
from typing import Dict, Tuple, List, Callable, Sequence

import numpy as np
import pandas as pd

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary

from mfsindy.experiments import (
    EnsembleConfigMixin,
    IntraTrajectoryGLSData,
    MonteCarloConfig,
    MultiTrajectoryGLSData,
    ValidationSupport,
    WeakValidationBlock,
    build_polynomial_rollout_models,
    coefficient_errors,
    fit_multi_trajectory_weak_gls_models,
    run_intra_trajectory_gls_experiment,
    run_monte_carlo_experiment,
    run_multi_trajectory_gls_experiment,
    select_validation_support,
)
from mfsindy.weighted_weak_pde_library import (
    DedupedWeakPDELibrary,
    WeightedWeakPDELibrary,
    weak_design_report,
    weak_validity_ratio,
)

PENDULUM_STATE_NAMES = ("theta", "omega")

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
    T_train: float = 1.0
    T_true: float = 10.0

    # physical parameters
    g: float = 9.81
    L: float = 1.0
    c: float = 0.5

    # SINDy settings
    poly_degree: int = 1
    H_xt: float | None = None
    K: int | None = None
    deduplicate: bool = True   # drop test functions whose support duplicates another's
    p: int | None = None
    stlsq_threshold: float = 0.1
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
        metadata={"t_grid": t_train, "weak_seed": cfg.seed_base + 10_000 + run_idx},
    )


def _pendulum_library(batch: MultiTrajectoryGLSData, cfg: PendulumMultiTrajectoryGLSConfig):
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )
    weak_kwargs = {}
    if cfg.K is not None:
        weak_kwargs["K"] = cfg.K
    if cfg.H_xt is not None:
        weak_kwargs["H_xt"] = cfg.H_xt
    if cfg.p is not None:
        weak_kwargs["p"] = cfg.p

    return WeakPDELibrary(
        function_library=base_library,
        spatiotemporal_grid=batch.metadata["t_grid"],
        **weak_kwargs,
    )


def _pendulum_true_coefficients(_: MultiTrajectoryGLSData, cfg: PendulumMultiTrajectoryGLSConfig) -> np.ndarray:
    return build_true_pendulum_coefficients(g=cfg.g, L=cfg.L, c=cfg.c)


def _pendulum_make_weak_library(
    batch: MultiTrajectoryGLSData,
    cfg: PendulumMultiTrajectoryGLSConfig,
    *,
    variance_field: np.ndarray | None,
    whitener_mode: str = "full",
):
    np.random.seed(int(batch.metadata["weak_seed"]))
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )
    t_grid = batch.metadata["t_grid"]
    common_kwargs = {
        "function_library": base_library,
        "spatiotemporal_grid": t_grid,
    }
    t_values = np.asarray(t_grid, dtype=float).ravel()
    extent = float(t_values.max() - t_values.min())
    H = cfg.H_xt if cfg.H_xt is not None else extent / 20.0
    common_kwargs["H_xt"] = H
    if cfg.p is not None:
        common_kwargs["p"] = cfg.p
    if cfg.K is not None:
        K_requested = int(cfg.K)
    else:
        # Coverage 2: every point lies under two test functions on average.
        # Coverage and support width are separable knobs. Coverage sets the
        # conditioning of the weak covariance -- 1 gives cond ~1e1, 2 ~1e2,
        # 10 ~1e7 -- while the support sets kappa, the validity of the
        # covariance model. Two keeps both in hand: it doubles the number of
        # weak equations over a bare tiling while leaving cond in the hundreds.
        K_requested = max(2, int(round(extent / H)))
    common_kwargs["K"] = K_requested

    if variance_field is None:
        return DedupedWeakPDELibrary(deduplicate=cfg.deduplicate, **common_kwargs)
    return WeightedWeakPDELibrary(
        spatiotemporal_weights=variance_field,
        whitener_mode=whitener_mode,
        deduplicate=cfg.deduplicate,
        **common_kwargs,
    )


def _pendulum_fit_multi_trajectory_weak_gls_models(
    batch: MultiTrajectoryGLSData,
    cfg: PendulumMultiTrajectoryGLSConfig,
    optimizer_factory,
    *,
    t_argument,
    noise_hf_abs: float,
    noise_lf_abs: float,
    methods: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    del t_argument

    def weak_block_builder(
        traj: np.ndarray,
        variance_field: np.ndarray | None,
        *,
        whitener_mode: str = "full",
    ):
        lib = _pendulum_make_weak_library(
            batch, cfg, variance_field=variance_field, whitener_mode=whitener_mode
        )
        theta = np.asarray(lib.fit_transform([traj])[0])
        rhs = np.asarray(lib.convert_u_dot_integral(traj))
        return theta, rhs

    return fit_multi_trajectory_weak_gls_models(
        batch,
        optimizer_factory,
        weak_block_builder=weak_block_builder,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
        methods=methods,
    )


def _pendulum_jacobian_norm(trajectory: np.ndarray, cfg) -> np.ndarray:
    """|grad F| along the trajectory. The linearised pendulum is linear, so this
    is constant, but it is returned per sample for a common interface."""

    n = np.asarray(trajectory, dtype=float).shape[0]
    J = np.array([[0.0, 1.0], [-cfg.g / cfg.L, -cfg.c]])
    return np.full(n, float(np.linalg.norm(J, ord=2)))


def pendulum_weak_design(
    cfg: PendulumMultiTrajectoryGLSConfig,
    *,
    run_idx: int = 0,
) -> dict:
    """The weak design a config actually produces, for the record in the paper.

    K is requested from the support width, then clamped to the rank the design
    supports and stripped of duplicate supports, so the requested count is not
    what gets fitted. This builds the library one config would build and reports
    the realised numbers, with the conditioning of the weak covariance.
    """

    state_std = _pendulum_reference_state_std(cfg)
    noise_hf_abs = cfg.noise_hf_rel * state_std
    batch = _pendulum_batch(run_idx, cfg, noise_hf_abs, cfg.noise_lf_rel * state_std)

    t_values = np.asarray(batch.metadata["t_grid"], dtype=float).ravel()
    extent = float(t_values.max() - t_values.min())
    H = cfg.H_xt if cfg.H_xt is not None else extent / 20.0
    K_requested = int(cfg.K) if cfg.K is not None else max(2, int(round(extent / H)))

    variance_field = np.full(batch.hf[0].shape[:-1], noise_hf_abs**2, dtype=float)
    library = _pendulum_make_weak_library(batch, cfg, variance_field=variance_field)
    library.fit_transform([batch.hf[0]])

    report = weak_design_report(library, K_requested)
    # The covariance model keeps only the derivative-driven term and drops the
    # one carrying the library Jacobian; kappa is the ratio of the two, and the
    # approximation holds where it is small.
    kappa = weak_validity_ratio(library, _pendulum_jacobian_norm(batch.hf[0], cfg))
    report["kappa_median"] = float(np.median(kappa))
    report["kappa_max"] = float(kappa.max())
    return report


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
        fit_models_fn=_pendulum_fit_multi_trajectory_weak_gls_models,
        coef_postprocess=lambda arr: arr.T,
        progress_desc="Monte Carlo pendulum MF",
    )


def fit_pendulum_multi_trajectory_rollout_models(
    cfg: PendulumMultiTrajectoryGLSConfig,
    *,
    hf_trajectories: list[np.ndarray],
    lf_trajectories: list[np.ndarray],
    t_grid: np.ndarray,
    weak_seed: int | None = None,
) -> dict[str, Any]:
    """Fit HF/LF/MF/MF_w pendulum models and wrap them for rollout validation."""

    if not hf_trajectories and not lf_trajectories:
        raise ValueError("At least one HF or LF trajectory is required.")

    state_std = _pendulum_reference_state_std(cfg)
    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std
    batch = MultiTrajectoryGLSData(
        hf=hf_trajectories,
        lf=lf_trajectories,
        t_argument=cfg.dt,
        metadata={
            "t_grid": np.asarray(t_grid, dtype=float),
            "weak_seed": cfg.seed_base if weak_seed is None else int(weak_seed),
        },
    )
    coef_map = _pendulum_fit_multi_trajectory_weak_gls_models(
        batch,
        cfg,
        cfg.make_optimizer,
        t_argument=cfg.dt,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
    )
    reference_trajectory = hf_trajectories[0] if hf_trajectories else lf_trajectories[0]
    return build_polynomial_rollout_models(
        coef_map,
        poly_degree=cfg.poly_degree,
        reference_trajectory=reference_trajectory,
        state_names=PENDULUM_STATE_NAMES,
        include_bias=False,
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
    c: float = 0.5

    # heteroscedastic noise model: sigma(t) = sigma0 + alpha * |omega(t)|
    sigma0: float = 0.0
    alpha: float = 0.15

    # weak-library settings
    poly_degree: int = 1
    derivative_order: int = 1
    H_xt: float | None = None
    K: int | None = None            # derived from H_xt when None
    p: int | None = None
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
    )


def build_pendulum_weak_validation_blocks(
    cfg: PendulumMultiTrajectoryGLSConfig,
    trajectories: list[np.ndarray],
    *,
    t_grid: np.ndarray,
    H_val=None,
    weak_seed: int | None = None,
    group_name: str = "validation",
) -> list[WeakValidationBlock]:
    """Held-out weak-form blocks for pendulum trajectories.

    ``H_val`` fixes the support of the validation system independently of the
    candidate being scored. Without it the target moves with the candidate --
    a different Theta, a different b and a different R^2 denominator per grid
    point -- so the scores would not be comparable across the grid, which is
    the whole purpose of scoring them. ``K`` is left to follow ``H_val``.
    """

    val_cfg = cfg if H_val is None else replace(cfg, H_xt=H_val, K=None)
    t_grid = np.asarray(t_grid, dtype=float)
    seed = cfg.seed_base if weak_seed is None else int(weak_seed)

    blocks: list[WeakValidationBlock] = []
    for traj_idx, trajectory in enumerate(trajectories):
        batch = MultiTrajectoryGLSData(
            hf=[trajectory],
            lf=[],
            t_argument=cfg.dt,
            metadata={"t_grid": t_grid, "weak_seed": seed},
        )
        library = _pendulum_make_weak_library(batch, val_cfg, variance_field=None)
        theta = np.asarray(library.fit_transform([trajectory])[0])
        rhs = np.asarray(library.convert_u_dot_integral(trajectory))
        blocks.append(
            WeakValidationBlock(
                theta=theta, rhs=rhs, group=group_name, trajectory=traj_idx, block=0
            )
        )
    return blocks


def pendulum_validation_support(
    cfg: PendulumMultiTrajectoryGLSConfig,
    validation_trajectory: np.ndarray,
    *,
    t_grid: np.ndarray,
    sigma: float,
    candidates,
    weak_seed: int | None = None,
    min_ceiling: float = 0.99,
    max_kappa: float = float("inf"),
    min_rows: int | None = None,
) -> ValidationSupport:
    """Choose the validation support for pendulum by the stated rule.

    The noise ceiling on ``b`` rejects the narrow supports and the smallest
    survivor wins, since rows are what let the metric tell candidates apart.
    Kappa is reported alongside but does not bind: it governs the covariance
    model, which this system never invokes -- the validation library is built
    unweighted and scored with an unweighted R^2. Kappa belongs to the rungs
    that whiten (VHF, VLF, PMF, VMF, MF_w), on their fitting support.
    See :func:`mfsindy.experiments.select_validation_support`.
    """

    t_grid = np.asarray(t_grid, dtype=float)
    seed = cfg.seed_base if weak_seed is None else int(weak_seed)
    jacobian_norm = _pendulum_jacobian_norm(validation_trajectory, cfg)

    def build(H):
        batch = MultiTrajectoryGLSData(
            hf=[validation_trajectory],
            lf=[],
            t_argument=cfg.dt,
            metadata={"t_grid": t_grid, "weak_seed": seed},
        )
        library = _pendulum_make_weak_library(
            batch, replace(cfg, H_xt=H, K=None), variance_field=None
        )
        library.fit_transform([validation_trajectory])
        return library, np.asarray(library.convert_u_dot_integral(validation_trajectory))

    return select_validation_support(
        build,
        candidates=candidates,
        sigma=sigma,
        kappa_fn=lambda library, _H: float(
            np.median(weak_validity_ratio(library, jacobian_norm))
        ),
        min_ceiling=min_ceiling,
        max_kappa=max_kappa,
        min_rows=min_rows,
    )


def pendulum_kappa_by_support(
    cfg: PendulumMultiTrajectoryGLSConfig,
    candidates,
    *,
    n_reference: int = 10,
    weak_seed: int | None = None,
) -> pd.DataFrame:
    """Kappa per candidate fitting support, on fixed reference trajectories.

    Kappa depends on the trajectory through ``|grad F(u)|``, so computing it on
    whichever draw the Monte Carlo happened to produce makes the admissible set
    a function of the seed -- on Lorenz the support at T/20 straddles 0.25
    between draws. Clean reference trajectories at the training horizon, drawn
    from ``cfg.seed_base``, keep it reproducible, and taking the worst over
    several of them keeps it from turning on one lucky draw.

    ``kappa_median`` -- the median test function, median reference trajectory --
    is the gate statistic, and ``kappa_lo``/``kappa_hi`` bracket it with the
    best and worst reference. The spread is wide where the bound actually
    matters: on Lorenz at T/20 the median is 0.23 but individual trajectories
    run from 0.17 to 0.44, because ``|grad F|`` varies that much over the
    attractor. Taking the worst reference instead would make the admissible set
    depend on how many references were drawn and from which seed -- the same
    support flips either side of 0.25 between ``seed_base`` 0 and 1234 -- so the
    median is used and the range is reported rather than buried. The bound is a
    guideline, not a theorem, and a support sitting near it is marginal rather
    than disqualified; the columns are there so that shows.

    ``kappa_max`` is the strict reading of (A.3), the worst test function on the
    worst reference, kept for contrast.
    """

    seed = cfg.seed_base if weak_seed is None else int(weak_seed)
    rows: list[dict] = []
    for H in candidates:
        per_traj_max: list[float] = []
        per_traj_median: list[float] = []
        K_used = 0
        for j in range(int(n_reference)):
            trajectories, t_grid, _ = generate_pendulum_dataset(
                n_traj=1,
                T=cfg.T_train,
                dt=cfg.dt,
                noise_level=0.0,
                seed=cfg.seed_base + j,
                g=cfg.g,
                L=cfg.L,
                c=cfg.c,
            )
            trajectory = trajectories[0]
            batch = MultiTrajectoryGLSData(
                hf=[trajectory],
                lf=[],
                t_argument=cfg.dt,
                metadata={"t_grid": t_grid, "weak_seed": seed},
            )
            library = _pendulum_make_weak_library(
                batch, replace(cfg, H_xt=H, K=None), variance_field=None
            )
            library.fit_transform([trajectory])
            ratios = weak_validity_ratio(
                library, _pendulum_jacobian_norm(trajectory, cfg)
            )
            per_traj_max.append(float(np.max(ratios)))
            per_traj_median.append(float(np.median(ratios)))
            K_used = int(library.K)
        rows.append(
            {
                "H_xt": H,
                "K": K_used,
                "kappa_median": float(np.median(per_traj_median)),
                "kappa_lo": float(np.min(per_traj_median)),
                "kappa_hi": float(np.max(per_traj_median)),
                "kappa_max": float(np.max(per_traj_max)),
            }
        )
    return pd.DataFrame(rows)
