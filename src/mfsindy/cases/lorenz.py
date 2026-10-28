# ---------------------------------------------------------------------------
# Lorenz system: generators and true coefficients
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Sequence

import inspect

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import pysindy as ps

from mfsindy.experiments import (
    EnsembleConfigMixin,
    IntraTrajectoryGLSData,
    MonteCarloConfig,
    MultiTrajectoryGLSData,
    fit_multi_trajectory_weak_gls_models,
    run_intra_trajectory_gls_experiment,
    run_multi_trajectory_gls_experiment,
)
from mfsindy.weighted_weak_pde_library import (
    DedupedWeakPDELibrary,
    WeightedWeakPDELibrary,
    weak_design_report,
    weak_validity_ratio,
)

LORENZ_STATE_NAMES = ("x", "y", "z")


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
class LorenzMultiTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the Lorenz multi-fidelity SINDy experiment."""

    # multi-fidelity settings
    n_lf: int = 10
    n_hf: int = 1

    # relative noise levels (wrt std of reference trajectory)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # time discretization
    dt: float = 1e-3
    T_train: float = 1.0
    T_true: float = 100.0
    T_forecast: float = 2.0

    # SINDy settings
    poly_degree: int = 2
    H_xt: float | None = None
    K: int | None = None            # derived from H_xt when None
    deduplicate: bool = True   # drop test functions whose support duplicates another's
    p: int | None = None
    # pysindy's STLSQ defaults to alpha=0.05, a ridge penalty that is not scale
    # aware. The weak targets here are small -- |b| is order 10 per state -- so
    # a fixed penalty on coefficients of order 10 biases the fit badly: at 0.05
    # a clean HF fit returns 9 terms with MAE 0.52 against the truth, at 1e-12
    # it returns exactly the 7 true terms with MAE 0.0094.
    #
    # It also falls unevenly across the rungs. Whitening multiplies Theta and b
    # by roughly 1/sigma, so the weighted rungs feel a fixed ridge far less than
    # the unweighted ones, which would flatter the comparison the paper draws.
    n_ensemble_models: int = 100

    # random seeds
    seed_base: int = 0
    seed_forecast_ic: int = 999

    # output
    results_filename: str = "lorenz_mf_errors.csv"


def _lorenz_reference_state_std(cfg: LorenzMultiTrajectoryGLSConfig) -> float:
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
    cfg: LorenzMultiTrajectoryGLSConfig,
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
        metadata={"t_grid": t_train, "weak_seed": cfg.seed_base + 10_000 + run_idx},
    )


def _lorenz_true_coefficients(_: MultiTrajectoryGLSData, cfg: LorenzMultiTrajectoryGLSConfig) -> np.ndarray:
    return build_true_coefficient_matrix()


def _lorenz_make_weak_library(
    batch: MultiTrajectoryGLSData,
    cfg: LorenzMultiTrajectoryGLSConfig,
    *,
    variance_field: np.ndarray | None,
    whitener_mode: str = "full",
):
    np.random.seed(int(batch.metadata["weak_seed"]))
    poly_lib = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )
    t_grid = batch.metadata["t_grid"]
    common_kwargs = {
        "function_library": poly_lib,
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


def _lorenz_fit_multi_trajectory_weak_gls_models(
    batch: MultiTrajectoryGLSData,
    cfg: LorenzMultiTrajectoryGLSConfig,
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
        return _lorenz_make_weak_library(
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


def run_lorenz_multi_trajectory_gls_experiment(
    cfg: LorenzMultiTrajectoryGLSConfig,
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
        true_coefficients=_lorenz_true_coefficients,
        optimizer_factory=cfg.make_optimizer,
        fit_models_fn=_lorenz_fit_multi_trajectory_weak_gls_models,
        coef_postprocess=lambda arr: arr.T,
        progress_desc="Monte Carlo Lorenz MF",
    )


def _lorenz_jacobian_norm(trajectory: np.ndarray, cfg=None) -> np.ndarray:
    """|grad F| along the trajectory, for the covariance-validity ratio.

    The system parameters are not config fields; they are the defaults of
    ``lorenz``, so they are read from its signature rather than repeated here.
    """

    del cfg
    defaults = inspect.signature(lorenz).parameters
    sigma = float(defaults["sigma"].default)
    rho = float(defaults["rho"].default)
    beta = float(defaults["beta"].default)

    x, y, z = (np.asarray(trajectory, dtype=float)[:, i] for i in range(3))
    n = x.size
    J = np.zeros((n, 3, 3))
    J[:, 0, 0] = -sigma
    J[:, 0, 1] = sigma
    J[:, 1, 0] = rho - z
    J[:, 1, 1] = -1.0
    J[:, 1, 2] = -x
    J[:, 2, 0] = y
    J[:, 2, 1] = x
    J[:, 2, 2] = -beta
    return np.linalg.norm(J, ord=2, axis=(1, 2))


def lorenz_weak_design(
    cfg: LorenzMultiTrajectoryGLSConfig,
    *,
    run_idx: int = 0,
) -> dict:
    """The weak design a config actually produces, for the record in the paper.

    K is requested from the support width, then clamped to the rank the design
    supports and stripped of duplicate supports, so the requested count is not
    what gets fitted. This builds the library one config would build and reports
    the realised numbers, along with the conditioning of the weak covariance.
    """

    state_std = _lorenz_reference_state_std(cfg)
    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std
    batch = _lorenz_batch(run_idx, cfg, noise_hf_abs, noise_lf_abs)

    t_values = np.asarray(batch.metadata["t_grid"], dtype=float).ravel()
    extent = float(t_values.max() - t_values.min())
    H = cfg.H_xt if cfg.H_xt is not None else extent / 20.0
    K_requested = (
        int(cfg.K) if cfg.K is not None else max(2, int(round(extent / H)))
    )

    variance_field = np.full(batch.hf[0].shape[:-1], noise_hf_abs**2, dtype=float)
    library = _lorenz_make_weak_library(batch, cfg, variance_field=variance_field)
    library.fit_transform([batch.hf[0]])

    report = weak_design_report(library, K_requested)
    # Kappa is not reported here. It was, computed on this one noisy HF
    # trajectory, which disagreed with lorenz_kappa_by_support -- the median over
    # clean reference trajectories -- by up to 1.6x at the same support. One
    # quantity, one estimator: use the kappa table.
    return report


# ---------------------------------------------------------------------------
# Lorenz heteroscedastic GLS experiment (weak / weighted-weak SINDy)
# ---------------------------------------------------------------------------

@dataclass
class LorenzIntraTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the heteroscedastic Lorenz GLS experiment."""

    # time discretisation
    t0: float = 0.0
    t1: float = 10.0
    dt: float = 1e-3

    # heteroscedastic noise level: sigma(t) = noise_level * |U(t)| over the
    # components noise_model names (see lorenz_noise_variance)
    noise_level: float = 0.25
    noise_model: str = "xy"

    # weak-library settings
    poly_degree: int = 2
    derivative_order: int = 1
    H_xt: float = 0.005
    K: int | None = None                      # derived from H_xt when None
    p: int = 2
    include_bias: bool = False

    # SINDy / optimizer settings
    n_ensemble_models: int = 100

    # random seeds
    seed_base: int = 0

    # output
    results_filename: str = "lorenz_weighted_errors.csv"


def lorenz_noise_variance(U_clean: np.ndarray, cfg) -> np.ndarray:
    """Pointwise noise variance along a clean Lorenz trajectory.

    ``"xy"`` (the default) scales the noise with the distance from the z axis,
    ``"norm"`` with the full state norm. The difference is whether the
    trajectory has clean regions. Under ``"norm"`` the relative noise
    sigma/|U| is exactly ``noise_level`` everywhere, so no stretch of the
    trajectory is quieter than any other relative to its own signal, and a
    weighting has nothing to exploit. Under ``"xy"`` it falls towards zero
    wherever the trajectory crosses between the lobes, while z -- and so the
    dynamics -- stays large there. ``"norm"`` is kept as the counter-case.
    """

    if cfg.noise_model == "norm":
        magnitude = np.linalg.norm(U_clean, axis=1)
    elif cfg.noise_model == "xy":
        magnitude = np.linalg.norm(U_clean[:, :2], axis=1)
    else:
        raise ValueError(f"noise_model must be 'norm' or 'xy', got {cfg.noise_model!r}.")
    return np.maximum((cfg.noise_level * magnitude) ** 2, 1e-8)


def _make_lorenz_gls_artifacts(
    run_idx: int,
    cfg: LorenzIntraTrajectoryGLSConfig,
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

    # 2) Heteroscedastic noise, as lorenz_noise_variance defines it
    variance = lorenz_noise_variance(U_clean, cfg)
    std = np.sqrt(variance)
    noise = std[:, None] * rng.standard_normal(size=U_clean.shape)
    U_noisy = U_clean + noise

    # 3) Spatiotemporal grid and polynomial library
    XT = t_eval[:, None]
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=cfg.include_bias,
    )

    # K and the support width are resolved exactly as in Part I: the support sets
    # the count at coverage 2, so a wider test function gets proportionally fewer.
    # Part II previously asked for five times as many, of which a third were
    # duplicates of each other.
    t_values = np.asarray(XT, dtype=float).ravel()
    extent = float(t_values.max() - t_values.min())
    H_weak = cfg.H_xt if cfg.H_xt is not None else extent / 20.0
    if cfg.K is not None:
        K_weak = int(cfg.K)
    else:
        K_weak = max(2, int(round(extent / H_weak)))

    tf_seed = cfg.seed_base + 1000 + run_idx

    # 3a) Unweighted weak library
    np.random.seed(tf_seed)
    weak_lib = DedupedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        is_uniform=True,
        K=K_weak,
        p=cfg.p,
        H_xt=H_weak,
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
        K=K_weak,
        p=cfg.p,
        H_xt=H_weak,
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
        K=K_weak,
        p=cfg.p,
        H_xt=H_weak,
        include_bias=cfg.include_bias,
    )

    C_true = build_true_coefficient_matrix()


    # Same heteroscedastic covariance, diagonal only: each weak equation is
    # weighted by its own variance, with the correlations between overlapping
    # test functions discarded. In Part I this rung collapses to one scalar per
    # group, because the variance is constant along a trajectory there. Here it
    # does not -- the diagonal spans a factor of ~50 -- so this is what isolates
    # the off-diagonal contribution from the variance weighting.
    np.random.seed(tf_seed)
    weighted_weak_lib_var_diag = WeightedWeakPDELibrary(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        spatiotemporal_weights=weights_scaled,
        is_uniform=True,
        K=K_weak,
        p=cfg.p,
        H_xt=H_weak,
        include_bias=cfg.include_bias,
        whitener_mode="diag",
    )
    libraries = {
        "No weighting": weak_lib,
        "Variance GLS": weighted_weak_lib_var,
        "Ones GLS": weighted_weak_lib_ones,
        "Variance diag": weighted_weak_lib_var_diag,
    }

    return IntraTrajectoryGLSData(
        data=U_noisy,
        t_argument=t_eval,
        libraries=libraries,
        true_coefficients=C_true,
    )


def run_lorenz_intra_trajectory_gls_experiment(
    cfg: LorenzIntraTrajectoryGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic Lorenz GLS experiment.
    """
    def builder(run_idx: int, cfg: LorenzIntraTrajectoryGLSConfig) -> IntraTrajectoryGLSData:
        # Seeded by run, not drawn from one stream: the runner builds a run once
        # per candidate support, and every support must see the same data.
        return _make_lorenz_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx))

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="Monte Carlo Lorenz GLS",
        coef_postprocess=lambda coef, _method: np.asarray(coef).T,
    )


def lorenz_kappa_by_support(
    cfg: LorenzMultiTrajectoryGLSConfig,
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
            trajectories, t_grid, _ = generate_lorenz_dataset(
                n_traj=1,
                T=cfg.T_train,
                dt=cfg.dt,
                noise_level=0.0,
                seed=cfg.seed_base + j,
            )
            trajectory = trajectories[0]
            batch = MultiTrajectoryGLSData(
                hf=[trajectory],
                lf=[],
                t_argument=cfg.dt,
                metadata={"t_grid": t_grid, "weak_seed": seed},
            )
            library = _lorenz_make_weak_library(
                batch, replace(cfg, H_xt=H, K=None), variance_field=None
            )
            library.fit_transform([trajectory])
            ratios = weak_validity_ratio(
                library, _lorenz_jacobian_norm(trajectory, cfg)
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
