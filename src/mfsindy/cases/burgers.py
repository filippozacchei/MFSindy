# utils.py
"""
Utilities for 1D viscous Burgers experiments:

- Finite-difference Burgers solver (periodic BC)
- Random Gaussian-bump initial conditions
- Heteroscedastic noise sigma = a s + b |u| (Part II, as in the ODE cases)
- Weak / weighted-weak libraries (WeakPDELibrary, WeightedWeakPDELibrary)
- Monte Carlo GLS experiment (heteroscedastic noise)
- Multi-fidelity data generation and PDE-SINDy models (HF / LF / MF / MF_w)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Sequence

import numpy as np
import pandas as pd

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
    pde_scale_separation_ratio,
    pde_sensitivity_fields,
    pde_weak_validity_ratio,
    weak_design_report,
)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
from ..experiments.artifacts import build_pde_intra_trajectory_artifacts

@dataclass
class BurgersConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration for the 1D Burgers GLS experiment (heteroscedastic noise)."""

    n_runs: int = 100

    # Domain and discretisation
    L: float = 8.0          # half-domain, domain [-L, L]
    NX: int = 32            # measured points; solved on NX * SOLVER_REFINE
    t0: float = 0.0
    t1: float = 1.0
    n_traj: int = 10        # fields per run, each from its own initial condition
    dt: float = 1e-3        # the Part II step shared with the ODE cases; also the
                            # explicit-Euler step of burgers_solver
    nu: float = 0.1         # viscosity, as in Part I

    # pointwise noise, as in every Part II case (see burgers_noise_variance):
    # sigma(x, t) = noise_floor * s + noise_level * |u(x, t)|
    noise_level: float = 0.15
    noise_floor: float = 1e-4

    # Weak-library settings
    poly_degree: int = 1
    derivative_order: int = 2
    # Per-dimension test-function support [h_x, h_t]. Part II used to leave this
    # to pysindy and pin K at 100, which is a different coverage at every
    # support and so not a basis for comparing supports; K now follows H_xt at
    # coverage 2, as in Part I, unless it is set explicitly.
    H_xt: list[float] | None = None
    K: int | None = None
    p: int = 2                 # as every other case; unset, pysindy's default is 4
    include_bias: bool = True

    # SINDy / optimizer settings
    n_ensemble_models: int = 100

    # Output
    results_filename: str = "burgers_weighted_errors.csv"


@dataclass
class BurgersMultiTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Configuration container for the Burgers multi-fidelity experiment."""

    # spatial / temporal discretization
    L: float = 8.0
    NX: int = 32            # measured points; solved on NX * SOLVER_REFINE
    dt: float = 1e-3
    T_train: float = 1.0
    nu: float = 0.1

    # multi-fidelity settings (numbers of trajectories)
    n_lf: int = 10
    n_hf: int = 1
    H_xt: list[float] | None = None
    K: int | None = None       # derived from H_xt when None
    deduplicate: bool = True   # drop test functions whose support duplicates another's
    p: int | None = None       # test-function polynomial degree

    # relative noise levels (wrt state std)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # SINDy / PDE-library settings
    poly_degree: int = 1               # polynomial degree in u
    derivative_order: int = 2          # spatial derivative order
    include_interaction: bool = True
    include_bias: bool = True

    n_ensemble_models: int = 100

    # random seeds
    seed_base: int = 231

    # output
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
    _, t = make_space_time_grid(cfg)
    NX = u0.shape[0]
    dx = 2.0 * cfg.L / NX       # the grid of u0, which may be finer than cfg's
    NT = len(t)

    u = u0.copy()
    U = np.zeros((NT, NX))
    U[0] = u

    for n in range(1, NT):
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
        uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
        u = u + cfg.dt * (-u * ux + cfg.nu * uxx)
        U[n] = u

    return U


#: The fields are solved on a grid this many times finer in space and sampled
#: onto the measurement grid. Solved on the measurement grid itself, the data
#: is a solution of the finite-difference scheme rather than of the PDE, and the
#: true coefficients leave a residual of the size of the HF noise.
SOLVER_REFINE = 4


def solve_random_field(rng: np.random.Generator, cfg) -> np.ndarray:
    """One clean field ``(NT, NX)`` from a random bump, solved SOLVER_REFINE
    times finer in space and sampled onto cfg's grid."""

    fine = replace(cfg, NX=cfg.NX * SOLVER_REFINE)
    return burgers_solver(random_initial_condition(rng, fine), fine)[:, ::SOLVER_REFINE]


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


def _canonicalize_burgers_trajectory(
    data: np.ndarray,
    t_grid: np.ndarray,
    *,
    variance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return Burgers fields in canonical ``(n_x, n_t, 1)`` order."""

    data = np.asarray(data, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    if data.ndim != 3 or data.shape[-1] != 1:
        raise ValueError("data must have shape (n_x, n_t, 1) or (n_t, n_x, 1).")

    aligned_variance = None if variance is None else np.asarray(variance, dtype=float)
    if data.shape[1] == t_grid.shape[0]:
        U = data
    elif data.shape[0] == t_grid.shape[0]:
        U = np.swapaxes(data, 0, 1)
        if aligned_variance is not None:
            aligned_variance = aligned_variance.T
    else:
        raise ValueError("Could not align data with t_grid along the time axis.")

    return U, aligned_variance


# ---------------------------------------------------------------------------
# Single Monte Carlo run (heteroscedastic GLS experiment)
# ---------------------------------------------------------------------------

def burgers_noise_variance(U_clean: np.ndarray, cfg) -> np.ndarray:
    """Pointwise noise variance of one field, shape of ``U_clean``.

    sigma(x, t) = noise_floor * s + noise_level * |u(x, t)|: the two-component
    measurement-error model of the ODE cases (lorenz_noise_variance), with s the
    spread of the field over space and time, so the floor is in its units.
    """

    U_clean = np.asarray(U_clean, dtype=float)
    sigma = cfg.noise_floor * np.std(U_clean) + cfg.noise_level * np.abs(U_clean)
    return np.maximum(sigma ** 2, 1e-12)


def burgers_run_trajectories(cfg: BurgersConfig, rng: np.random.Generator):
    """The cfg.n_traj fields of one Part II run, as ``(x, t, clean, noisy, variance)``.

    Fields are ``(NX, NT, 1)`` and variances ``(NX, NT)``. Each field starts from
    its own random bump and gets its own noise, drawn from burgers_noise_variance.
    The Monte Carlo builder, main2.py and the notebook all draw through this
    function, so they show the same data.
    """

    x, t = make_space_time_grid(cfg)
    clean, noisy, variances = [], [], []
    for _ in range(cfg.n_traj):
        U_clean = solve_random_field(rng, cfg).T[:, :, None]
        variance = burgers_noise_variance(U_clean[:, :, 0], cfg)
        clean.append(U_clean)
        noisy.append(U_clean + np.sqrt(variance)[:, :, None] * rng.standard_normal(size=U_clean.shape))
        variances.append(variance)
    return x, t, clean, noisy, variances


def _make_burgers_gls_artifacts(
    run_idx: int,
    cfg: BurgersConfig,
    rng: np.random.Generator,
) -> list[IntraTrajectoryGLSData]:
    """The four weightings for each field of one Burgers GLS run.

    Every field and every rung is placed on the same test functions (one seed),
    so the rungs differ by covariance model alone. K follows the support at
    coverage 2 unless the config pins it. The variances are passed as they are,
    not rescaled per field: the fields' weak systems are stacked, and a
    per-field rescaling would change their relative weights. The inner
    polynomial library carries no bias because the weak library supplies the
    constant, and two of them would be the same column twice.
    """

    x, t, _, noisy, variances = burgers_run_trajectories(cfg, rng)
    X, T = np.meshgrid(x, t)
    XT = np.asarray([X, T]).T                # (NX, NT, 2)
    return [
        build_pde_intra_trajectory_artifacts(
            data,
            XT,
            t_grid=t,
            variance=variance,
            poly_degree=cfg.poly_degree,
            derivative_order=cfg.derivative_order,
            include_bias=cfg.include_bias,
            function_include_bias=False,
            K=cfg.K,
            H_xt=cfg.H_xt,
            coverage=cfg.coverage,
            p=cfg.p,
            weak_seed=cfg.seed_base + 1000 + run_idx,
            true_coefficients=build_true_burgers_coefficients(cfg.nu),
        )
        for data, variance in zip(noisy, variances)
    ]


def run_burgers_experiment(
    cfg: BurgersConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic GLS experiment.
    """
    def builder(run_idx: int, cfg: BurgersConfig) -> list[IntraTrajectoryGLSData]:
        # Seeded by run, not drawn from one stream: the runner builds a run once
        # per candidate support, and every support must see the same data.
        return _make_burgers_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx))

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="Monte Carlo Burgers GLS",
    )


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
    X_list : list of arrays (Nx, Nt, 1)
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
        U_clean = solve_random_field(rng, cfg_tmp).T  # (Nx, Nt)
        U = U_clean                                  # (Nx, Nt)

        noise = noise_level * rng.standard_normal(size=U.shape)
        U_noisy = U + noise
        X_list.append(U_noisy[:, :, None])           # (Nx, Nt, 1)

    return X_list, t, x, nu


def _burgers_reference_state_std(cfg: BurgersMultiTrajectoryGLSConfig) -> float:
    """Reference state std used to convert relative noise to absolute values."""

    X_ref_list, _, _, _ = generate_burgers_dataset(
        n_traj=1,
        T=cfg.T_train,
        dt=cfg.dt,
        L=cfg.L,
        NX=cfg.NX,
        nu=cfg.nu,
        noise_level=0.0,
        seed=cfg.seed_base,
    )
    return float(np.std(X_ref_list[0]))


def _burgers_batch(
    run_idx: int,
    cfg: BurgersMultiTrajectoryGLSConfig,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> MultiTrajectoryGLSData:
    """Return HF/LF training trajectories for a single Monte Carlo run."""

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
    return MultiTrajectoryGLSData(
        hf=X_hf,
        lf=X_lf,
        t_argument=cfg.dt,
        metadata={"t": t_train, "x": x_grid, "weak_seed": cfg.seed_base + 10_000 + run_idx},
    )


def _burgers_true_coefficients(_: MultiTrajectoryGLSData, cfg: BurgersMultiTrajectoryGLSConfig) -> np.ndarray:
    return build_true_burgers_coefficients(cfg.nu)


def _burgers_make_weak_library(
    batch: MultiTrajectoryGLSData,
    cfg: BurgersMultiTrajectoryGLSConfig,
    *,
    variance_field: np.ndarray | None,
    whitener_mode: str = "full",
):
    x = batch.metadata["x"]
    t = batch.metadata["t"]
    X, T = np.meshgrid(x, t)
    XT = np.asarray([X, T]).T
    np.random.seed(int(batch.metadata["weak_seed"]))
    base_library = ps.PolynomialLibrary(
        degree=cfg.poly_degree,
        include_bias=False,
    )
    common_kwargs = dict(
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=XT,
        is_uniform=True,
        include_bias=cfg.include_bias,
    )
    # Coverage 2: the test functions cover the grid twice over. Coverage sets
    # the conditioning of the weak covariance and the support sets kappa, so the
    # two are chosen independently.
    extents = (float(x.max() - x.min()), float(t.max() - t.min()))
    H = cfg.H_xt if cfg.H_xt is not None else [e / 20.0 for e in extents]
    common_kwargs["H_xt"] = H
    if cfg.p is not None:
        common_kwargs["p"] = cfg.p
    if cfg.K is not None:
        K_requested = int(cfg.K)
    else:
        domain = float(np.prod([2.0 * h for h in np.atleast_1d(H)]))
        K_requested = max(2, int(round(cfg.coverage * float(np.prod(extents)) / domain)))
    common_kwargs["K"] = K_requested
    if variance_field is None:
        return DedupedWeakPDELibrary(deduplicate=cfg.deduplicate, **common_kwargs)
    return WeightedWeakPDELibrary(
        spatiotemporal_weights=variance_field,
        whitener_mode=whitener_mode,
        deduplicate=cfg.deduplicate,
        **common_kwargs,
    )


def _burgers_fit_multi_trajectory_weak_gls_models(
    batch: MultiTrajectoryGLSData,
    cfg: BurgersMultiTrajectoryGLSConfig,
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
        return _burgers_make_weak_library(
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


def burgers_weak_design(
    cfg: BurgersMultiTrajectoryGLSConfig,
    *,
    weak_seed: int | None = None,
) -> dict:
    """The weak design a config actually produces, for the record in the paper.

    The test-function weights depend on the spatiotemporal grid alone, so this
    needs no solution of the PDE: a zero field on the right grid gives the same
    design the experiment builds. K is reported as requested and as used, since
    duplicate supports are dropped.
    """

    grid_cfg = BurgersConfig(L=cfg.L, NX=cfg.NX, t0=0.0, t1=cfg.T_train, dt=cfg.dt, nu=cfg.nu)
    x, t = make_space_time_grid(grid_cfg)
    field_shape = (x.size, t.size)

    batch = MultiTrajectoryGLSData(
        hf=[np.zeros(field_shape + (1,))],
        lf=[],
        t_argument=cfg.dt,
        metadata={
            "x": x,
            "t": t,
            "weak_seed": int(cfg.seed_base if weak_seed is None else weak_seed),
        },
    )
    extents = (float(x.max() - x.min()), float(t.max() - t.min()))
    H = cfg.H_xt if cfg.H_xt is not None else [e / 20.0 for e in extents]
    if cfg.K is not None:
        K_requested = int(cfg.K)
    else:
        domain = float(np.prod([2.0 * h for h in np.atleast_1d(H)]))
        K_requested = max(2, int(round(cfg.coverage * float(np.prod(extents)) / domain)))

    variance_field = np.ones(field_shape, dtype=float)
    library = _burgers_make_weak_library(batch, cfg, variance_field=variance_field)
    library.fit_transform([batch.hf[0]])

    report = weak_design_report(library, K_requested)
    # For a PDE the neglected term is controlled by scale separation between
    # the temporal and spatial supports rather than by the Jacobian, so the
    # validity ratio is analytic: h_t^2 * h_x^(-2m).
    widths = list(np.atleast_1d(H))
    report["kappa_scale_sep"] = pde_scale_separation_ratio(
        widths[-1], widths[:-1], cfg.derivative_order
    )
    return report


def run_burgers_multi_trajectory_gls_experiment(
    cfg: BurgersMultiTrajectoryGLSConfig,
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
    return run_multi_trajectory_gls_experiment(
        cfg,
        reference_state_std=_burgers_reference_state_std,
        dataset_builder=_burgers_batch,
        true_coefficients=_burgers_true_coefficients,
        optimizer_factory=cfg.make_optimizer,
        fit_models_fn=_burgers_fit_multi_trajectory_weak_gls_models,
        progress_desc="Monte Carlo Burgers MF",
    )


# ---------------------------------------------------------------------------
# Helper: get one set of GLS coefficients for animation / forecasting
# ---------------------------------------------------------------------------


def burgers_kappa_by_support(
    cfg: BurgersMultiTrajectoryGLSConfig,
    candidates,
    *,
    n_reference: int = 5,
    weak_seed: int | None = None,
    coefficients: np.ndarray | None = None,
) -> pd.DataFrame:
    """Kappa per candidate fitting support, on fixed reference trajectories.

    The PDE counterpart of the ODE tables, and computed the same way: from the
    assembled weak system, as a dimensionless ratio of squared norms. The older
    ``h_t^2 h_x^(-2m)`` form carries units, so it could be compared neither
    between benchmarks nor against the ODE kappa; this can.

    ``kappa_median`` is the reported statistic and ``kappa_lo``/``kappa_hi`` bracket
    it with the best and worst reference trajectory. References are clean and
    drawn from ``cfg.seed_base``, so the admissible set cannot move with the
    Monte Carlo draw.
    """

    seed = cfg.seed_base if weak_seed is None else int(weak_seed)
    if coefficients is None:
        coefficients = build_true_burgers_coefficients(nu=cfg.nu)

    references = []
    for j in range(int(n_reference)):
        trajectories, t_grid, x_grid, _ = generate_burgers_dataset(
            n_traj=1, T=cfg.T_train, dt=cfg.dt, noise_level=0.0,
            seed=cfg.seed_base + j, L=cfg.L, NX=cfg.NX, nu=cfg.nu,
        )
        references.append((np.asarray(trajectories[0]), t_grid, x_grid))

    rows: list[dict] = []
    for H in candidates:
        medians: list[float] = []
        maxima: list[float] = []
        K_used = 0
        for trajectory, t_grid, x_grid in references:
            batch = MultiTrajectoryGLSData(
                hf=[trajectory], lf=[], t_argument=cfg.dt,
                metadata={"t": t_grid, "x": x_grid, "weak_seed": seed},
            )
            library = _burgers_make_weak_library(
                batch, replace(cfg, H_xt=list(H), K=None), variance_field=None
            )
            library.fit_transform([trajectory])
            ratios = pde_weak_validity_ratio(
                library,
                pde_sensitivity_fields(library, trajectory, coefficients),
            )
            medians.append(float(np.median(ratios)))
            maxima.append(float(np.max(ratios)))
            K_used = int(library.K)
        rows.append(
            {
                "H_xt": list(H),
                "K": K_used,
                "kappa_median": float(np.median(medians)),
                "kappa_lo": float(np.min(medians)),
                "kappa_hi": float(np.max(medians)),
                "kappa_max": float(np.max(maxima)),
            }
        )
    return pd.DataFrame(rows)
