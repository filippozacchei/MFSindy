# ---------------------------------------------------------------------------
# Hopf oscillator: dynamics, trajectories, true coefficients
# ---------------------------------------------------------------------------

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
    fit_multi_trajectory_weak_gls_models,
    run_intra_trajectory_gls_experiment,
    run_multi_trajectory_gls_experiment,
    select_validation_support,
)
from mfsindy.weighted_weak_pde_library import (
    DedupedWeakPDELibrary,
    WeightedWeakPDELibrary,
    weak_design_report,
    weak_validity_ratio,
)

from scipy.integrate import solve_ivp  # at top of file if not already imported

HOPF_STATE_NAMES = ("x", "y")



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
class HopfMultiTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the Hopf multi-fidelity SINDy experiment."""

    # multi-fidelity settings
    n_lf: int = 10
    n_hf: int = 1

    # relative noise levels (wrt std of reference trajectory)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # time discretisation
    dt: float = 1e-3
    T_train: float = 1
    T_true: float = 10.0

    # Hopf parameters
    mu: float = 1.0
    omega: float = 1.0

    # SINDy settings
    poly_degree: int = 3
    H_xt: float | None = None
    K: int | None = None
    deduplicate: bool = True   # drop test functions whose support duplicates another's
    p: int | None = None
    stlsq_threshold: float = 0.5
    n_ensemble_models: int = 100

    # random seeds
    seed_base: int = 0

    # output
    results_filename: str = "hopf_mf_errors.csv"

def _hopf_reference_state_std(cfg: HopfMultiTrajectoryGLSConfig) -> float:
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
    cfg: HopfMultiTrajectoryGLSConfig,
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
        metadata={"t_grid": t_train, "weak_seed": cfg.seed_base + 10_000 + run_idx},
    )


def _hopf_library(batch: MultiTrajectoryGLSData, cfg: HopfMultiTrajectoryGLSConfig):
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


def _hopf_true_coefficients(_: MultiTrajectoryGLSData, cfg: HopfMultiTrajectoryGLSConfig) -> np.ndarray:
    return build_true_hopf_coefficients(mu=cfg.mu, omega=cfg.omega)


def _hopf_make_weak_library(
    batch: MultiTrajectoryGLSData,
    cfg: HopfMultiTrajectoryGLSConfig,
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


def _hopf_fit_multi_trajectory_weak_gls_models(
    batch: MultiTrajectoryGLSData,
    cfg: HopfMultiTrajectoryGLSConfig,
    optimizer_factory,
    *,
    t_argument,
    noise_hf_abs: float,
    noise_lf_abs: float,
    methods: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    del t_argument

    def weak_library_builder(
        variance_field: np.ndarray | None,
        *,
        whitener_mode: str = "full",
    ):
        return _hopf_make_weak_library(
            batch, cfg, variance_field=variance_field, whitener_mode=whitener_mode
        )

    return fit_multi_trajectory_weak_gls_models(
        batch,
        optimizer_factory,
        weak_library_builder=weak_library_builder,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
        methods=methods,
    )


def _hopf_jacobian_norm(trajectory: np.ndarray, cfg) -> np.ndarray:
    """|grad F| along the trajectory, for the covariance-validity ratio."""

    x, y = (np.asarray(trajectory, dtype=float)[:, i] for i in range(2))
    n = x.size
    J = np.zeros((n, 2, 2))
    J[:, 0, 0] = cfg.mu - 3.0 * x**2 - y**2
    J[:, 0, 1] = -cfg.omega - 2.0 * x * y
    J[:, 1, 0] = cfg.omega - 2.0 * x * y
    J[:, 1, 1] = cfg.mu - x**2 - 3.0 * y**2
    return np.linalg.norm(J, ord=2, axis=(1, 2))


def hopf_weak_design(
    cfg: HopfMultiTrajectoryGLSConfig,
    *,
    run_idx: int = 0,
) -> dict:
    """The weak design a config actually produces, for the record in the paper.

    K is requested from the support width, then clamped to the rank the design
    supports and stripped of duplicate supports, so the requested count is not
    what gets fitted. This builds the library one config would build and reports
    the realised numbers, with the conditioning of the weak covariance.
    """

    state_std = _hopf_reference_state_std(cfg)
    noise_hf_abs = cfg.noise_hf_rel * state_std
    batch = _hopf_batch(run_idx, cfg, noise_hf_abs, cfg.noise_lf_rel * state_std)

    t_values = np.asarray(batch.metadata["t_grid"], dtype=float).ravel()
    extent = float(t_values.max() - t_values.min())
    H = cfg.H_xt if cfg.H_xt is not None else extent / 20.0
    K_requested = int(cfg.K) if cfg.K is not None else max(2, int(round(extent / H)))

    variance_field = np.full(batch.hf[0].shape[:-1], noise_hf_abs**2, dtype=float)
    library = _hopf_make_weak_library(batch, cfg, variance_field=variance_field)
    library.fit_transform([batch.hf[0]])

    report = weak_design_report(library, K_requested)
    # The covariance model keeps only the derivative-driven term and drops the
    # one carrying the library Jacobian; kappa is the ratio of the two, and the
    # approximation holds where it is small.
    kappa = weak_validity_ratio(library, _hopf_jacobian_norm(batch.hf[0], cfg))
    report["kappa_median"] = float(np.median(kappa))
    report["kappa_max"] = float(kappa.max())
    return report


def run_hopf_multi_trajectory_gls_experiment(
    cfg: HopfMultiTrajectoryGLSConfig,
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
        fit_models_fn=_hopf_fit_multi_trajectory_weak_gls_models,
        coef_postprocess=lambda arr: arr.T,
        progress_desc="Monte Carlo Hopf MF",
    )


def fit_hopf_multi_trajectory_rollout_models(
    cfg: HopfMultiTrajectoryGLSConfig,
    *,
    hf_trajectories: list[np.ndarray],
    lf_trajectories: list[np.ndarray],
    t_grid: np.ndarray,
    weak_seed: int | None = None,
) -> dict[str, Any]:
    """Fit HF/LF/MF/MF_w Hopf models and wrap them for rollout validation."""

    if not hf_trajectories and not lf_trajectories:
        raise ValueError("At least one HF or LF trajectory is required.")

    state_std = _hopf_reference_state_std(cfg)
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
    coef_map = _hopf_fit_multi_trajectory_weak_gls_models(
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
        state_names=HOPF_STATE_NAMES,
        include_bias=False,
    )

# ---------------------------------------------------------------------------
# Hopf heteroscedastic GLS experiment (weak / weighted-weak SINDy)
# ---------------------------------------------------------------------------

@dataclass
class HopfIntraTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
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
    H_xt: float = 0.01
    K: int = int(5 * (t1-t0) / H_xt)
    p: int = 2
    include_bias: bool = False

    # SINDy / optimizer settings
    stlsq_threshold: float = 0.5
    n_ensemble_models: int = 20

    # output
    results_filename: str = "hopf_weighted_errors.csv"
    
def _build_hopf_gls_artifacts(
    run_idx: int,
    cfg: HopfIntraTrajectoryGLSConfig,
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


def run_hopf_intra_trajectory_gls_experiment(
    cfg: HopfIntraTrajectoryGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic Hopf GLS experiment.
    """
    rng = np.random.default_rng(cfg.seed_base)

    def builder(run_idx: int, cfg: HopfIntraTrajectoryGLSConfig) -> IntraTrajectoryGLSData:
        return _build_hopf_gls_artifacts(run_idx, cfg, rng)

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="Monte Carlo Hopf GLS",
        coef_postprocess=lambda coef, _method: np.asarray(coef).T,
    )


def build_hopf_weak_validation_blocks(
    cfg: HopfMultiTrajectoryGLSConfig,
    trajectories: list[np.ndarray],
    *,
    t_grid: np.ndarray,
    H_val=None,
    weak_seed: int | None = None,
    group_name: str = "validation",
) -> list[WeakValidationBlock]:
    """Held-out weak-form blocks for hopf trajectories.

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
        library = _hopf_make_weak_library(batch, val_cfg, variance_field=None)
        theta = np.asarray(library.fit_transform([trajectory])[0])
        rhs = np.asarray(library.convert_u_dot_integral(trajectory))
        blocks.append(
            WeakValidationBlock(
                theta=theta, rhs=rhs, group=group_name, trajectory=traj_idx, block=0
            )
        )
    return blocks


def hopf_validation_support(
    cfg: HopfMultiTrajectoryGLSConfig,
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
    """Choose the validation support for hopf by the stated rule.

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
    jacobian_norm = _hopf_jacobian_norm(validation_trajectory, cfg)

    def build(H):
        batch = MultiTrajectoryGLSData(
            hf=[validation_trajectory],
            lf=[],
            t_argument=cfg.dt,
            metadata={"t_grid": t_grid, "weak_seed": seed},
        )
        library = _hopf_make_weak_library(
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


def hopf_kappa_by_support(
    cfg: HopfMultiTrajectoryGLSConfig,
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
            trajectories, t_grid, _ = generate_hopf_dataset(
                n_traj=1,
                T=cfg.T_train,
                dt=cfg.dt,
                noise_level=0.0,
                seed=cfg.seed_base + j,
                mu=cfg.mu,
                omega=cfg.omega,
            )
            trajectory = trajectories[0]
            batch = MultiTrajectoryGLSData(
                hf=[trajectory],
                lf=[],
                t_argument=cfg.dt,
                metadata={"t_grid": t_grid, "weak_seed": seed},
            )
            library = _hopf_make_weak_library(
                batch, replace(cfg, H_xt=H, K=None), variance_field=None
            )
            library.fit_transform([trajectory])
            ratios = weak_validity_ratio(
                library, _hopf_jacobian_norm(trajectory, cfg)
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
