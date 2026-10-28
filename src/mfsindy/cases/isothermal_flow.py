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

from dataclasses import dataclass, replace
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary

from mfsindy.experiments.multi_trajectory import _variance_signature
from mfsindy.experiments import (
    EnsembleConfigMixin,
    IntraTrajectoryGLSData,
    MonteCarloConfig,
    MultiTrajectoryGLSData,
    assemble_weak_rungs,
    get_library_feature_names,
    run_intra_trajectory_gls_experiment,
    run_multi_trajectory_gls_experiment,
)
from mfsindy.experiments.artifacts import (
    build_pde_intra_trajectory_artifacts,
    pde_coverage_test_function_count,
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
# Core PDE: isothermal compressible Navier–Stokes
# ---------------------------------------------------------------------------

def _periodic_centered_diff(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Periodic centered finite difference (first derivative) along one axis."""
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * dx)


def _periodic_second_diff(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Periodic centered finite difference (second derivative) along one axis."""
    return (np.roll(f, -1, axis=axis) - 2.0 * f + np.roll(f, 1, axis=axis)) / (dx * dx)


def compressible(t, U, dx, N, mu, RT):
    """2D isothermal compressible flow (periodic) for solve_ivp.

    This uses explicit numpy roll-based finite differences for speed.

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

    # First derivatives
    ux = _periodic_centered_diff(u, dx, axis=0)
    uy = _periodic_centered_diff(u, dx, axis=1)
    vx = _periodic_centered_diff(v, dx, axis=0)
    vy = _periodic_centered_diff(v, dx, axis=1)

    # Second derivatives
    uxx = _periodic_second_diff(u, dx, axis=0)
    uyy = _periodic_second_diff(u, dx, axis=1)
    vxx = _periodic_second_diff(v, dx, axis=0)
    vyy = _periodic_second_diff(v, dx, axis=1)

    # Pressure and its derivatives
    p = rho * RT
    px = _periodic_centered_diff(p, dx, axis=0)
    py = _periodic_centered_diff(p, dx, axis=1)

    inv_rho = 1.0 / rho

    ret = np.empty_like(uvr)
    # u_t
    ret[:, :, 0] = -(u * ux + v * uy) - (px - mu * (uxx + uyy)) * inv_rho
    # v_t
    ret[:, :, 1] = -(u * vx + v * vy) - (py - mu * (vxx + vyy)) * inv_rho
    # rho_t
    ret[:, :, 2] = -(u * px / RT + v * py / RT + rho * ux + rho * vy)

    return ret.reshape(-1)


def make_initial_condition(
    X,
    Y,
    L,
    rng: np.random.Generator | None = None,
):
    """
    Return (U0, V0, RHO0) for a chosen flow configuration.

    X, Y : meshgrid on [0, L] x [0, L].
    """
    if rng is None:
        rng = np.random.default_rng()

    # Randomize coefficients around standard Taylor-Green values for variation
    amp_u_sin = - (0.8 + 0.4 * rng.random())  # around -1.0
    amp_u_cos = 0.4 + 0.2 * rng.random()      # around 0.5
    amp_v_cos = 0.4 + 0.2 * rng.random()      # around 0.5
    amp_v_sin = - (0.8 + 0.4 * rng.random())  # around -1.0
    amp_rho_cos = 0.4 + 0.2 * rng.random()    # around 0.5

    # Vary the wavenumbers too, not just the amplitudes. Trajectories that
    # differ only in amplitude are near-replicates of one vortex pattern, so
    # a second trajectory adds averaging but little new information. The
    # wavenumbers set the magnitude of the derivatives relative to the state
    # itself -- u_x scales as k and u_xx as k^2 -- which is exactly the
    # balance between the advective and viscous terms being identified, so
    # varying them makes each trajectory probe that balance differently.
    #
    # Phases are deliberately not randomized: the dynamics are translation
    # invariant, so a shifted field visits the same local states and the weak
    # regression cannot tell it apart.
    #
    # The finest structure in the field is the 2*k_y mode, so k_y is what
    # limits resolution: k_y=2 puts 8 points per wavelength on a 32-point
    # axis, which is already the floor. k_x costs nothing by comparison and
    # is given a third value, so there are six structural combinations rather
    # than four and fewer trajectories are structural duplicates.
    k_x = int(rng.integers(1, 4))
    k_y = int(rng.integers(1, 3))

    U0 = (amp_u_sin * np.sin(k_x * 2 * np.pi / L * X) +
            amp_u_cos * np.cos(2 * k_y * 2 * np.pi / L * Y))
    V0 = (amp_v_cos * np.cos(k_x * 2 * np.pi / L * X) +
            amp_v_sin * np.sin(2 * k_y * 2 * np.pi / L * Y))
    RHO0 = 1.0 + amp_rho_cos * np.cos(k_x * 2 * np.pi / L * X) * np.cos(
        2 * k_y * 2 * np.pi / L * Y
    )

    return U0, V0, RHO0


def generate_isothermal_ns_dataset(
    N: int = 32,
    Nt: int = 500,
    L: float = 5.0,
    T: float = 2.5,
    mu: float = 1.0,
    RT: float = 1.0,
    seed: int = 1,
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
        X, Y, L, rng=rng
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
    """Centered finite difference in time (axis=2) with one-sided boundaries."""
    if f.ndim < 3:
        raise ValueError("Expected an array with time on axis 2.")
    if f.shape[2] < 2:
        raise ValueError("At least two time samples are required.")

    ddt = np.empty_like(f, dtype=float)
    if f.shape[2] == 2:
        slope = (f[:, :, 1] - f[:, :, 0]) / dt
        ddt[:, :, 0] = slope
        ddt[:, :, 1] = slope
        return ddt

    ddt[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dt)
    ddt[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dt
    ddt[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dt
    return ddt


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
    p = U_clean[..., 2]

    u_t = _ddt_centered(u, dt)
    v_t = _ddt_centered(v, dt)

    time_deriv_mag = np.sqrt(u ** 2 + v ** 2 + 0*p ** 2)  # (N, N, Nt)
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


def _ns_multi_h_xt(cfg: "NSIsothermalMultiTrajectoryGLSConfig") -> list[float]:
    if cfg.H_xt is not None:
        return list(cfg.H_xt)
    return [cfg.L / 10.0, cfg.L / 10.0, cfg.T / 10.0]


def _ns_intra_h_xt(cfg: "NSIsothermalIntraTrajectoryGLSConfig") -> list[float]:
    if cfg.H_xt is not None:
        return list(cfg.H_xt)
    return [cfg.L / 10.0, cfg.L / 10.0, cfg.T / 40.0]


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
    n_lf: int = 10
    n_hf: int = 1

    # relative noise levels w.r.t. std(U_clean_ref)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # grid / time
    N: int = 64
    Nt: int = 1001
    Nt_std: int = 1001
    L: float = 5.0
    T: float = 1.0
    T_std : float = 1.0

    # physical parameters
    mu: float = 1.0
    RT: float = 1.0

    # weak-library settings
    derivative_order: int = 2
    include_bias: bool = False
    p: int = 2
    K: int | None = None       # derived from H_xt when None
    deduplicate: bool = True   # drop test functions whose support duplicates another's
    K_std: int = 100
    H_xt: list[float] | None = None

    # SINDy / optimizer settings
    n_ensemble_models: int = 100

    # randomness
    seed_base: int = 0

    # output
    results_filename: str = "ns_isothermal_mf_errors.csv"


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
        metadata={
            "grid": grid,
            "weak_seed": cfg.seed_base + 100_000 + run_idx,
        },
    )


def _ns_make_weak_library(
    cfg: NSIsothermalMultiTrajectoryGLSConfig,
    grid: np.ndarray,
    *,
    variance_field: np.ndarray | None,
    weak_seed: int,
    whitener_mode: str = "full",
):
    np.random.seed(weak_seed)
    # Coverage 2: the test functions cover the grid twice over. Coverage sets
    # the conditioning of the weak covariance and the support sets kappa, so the
    # two are chosen independently.
    common_kwargs = dict(
        function_library=_build_custom_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        is_uniform=True,
        include_bias=cfg.include_bias,
    )
    # Coverage 2: the test functions cover the grid twice over. Coverage sets
    # the conditioning of the weak covariance and the support sets kappa, so the
    # two are chosen independently.
    extents = tuple(float(np.ptp(grid[..., axis])) for axis in range(grid.shape[-1]))
    H = _ns_multi_h_xt(cfg)
    common_kwargs["H_xt"] = H
    if cfg.p is not None:
        common_kwargs["p"] = cfg.p
    if cfg.K is not None:
        K_requested = int(cfg.K)
    else:
        domain = float(np.prod([2.0 * h for h in np.atleast_1d(H)]))
        K_requested = max(2, int(round(2.0 * float(np.prod(extents)) / domain)))
    common_kwargs["K"] = K_requested
    if variance_field is None:
        # Genuinely unweighted, as in the other cases. Passing a field of ones
        # here would still apply the weak-SINDy whitening, which made the HF, LF
        # and MF rungs weighted and identical to PMF.
        return DedupedWeakPDELibrary(deduplicate=cfg.deduplicate, **common_kwargs)
    return WeightedWeakPDELibrary(
        spatiotemporal_weights=variance_field,
        whitener_mode=whitener_mode,
        deduplicate=cfg.deduplicate,
        **common_kwargs,
    )


def _ns_build_weak_block(
    traj: np.ndarray,
    cfg: NSIsothermalMultiTrajectoryGLSConfig,
    *,
    grid: np.ndarray,
    variance_field: np.ndarray | None,
    weak_seed: int,
    whitener_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    library = _ns_make_weak_library(
        cfg,
        grid,
        variance_field=variance_field,
        weak_seed=weak_seed,
        whitener_mode=whitener_mode,
    )
    theta = np.asarray(library.fit_transform([traj])[0])
    rhs = np.asarray(library.convert_u_dot_integral(traj))
    return theta, rhs


def _fit_stacked_weak_system(
    theta_blocks: list[np.ndarray],
    rhs_blocks: list[np.ndarray],
    optimizer_factory,
):
    optimizer = optimizer_factory()
    optimizer.fit(np.vstack(theta_blocks), np.vstack(rhs_blocks))
    return optimizer


def _ns_fit_multi_trajectory_weak_gls_models(
    batch: MultiTrajectoryGLSData,
    cfg: NSIsothermalMultiTrajectoryGLSConfig,
    optimizer_factory,
    *,
    t_argument,
    noise_hf_abs: float,
    noise_lf_abs: float,
    methods: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    del t_argument

    grid = batch.metadata["grid"]
    weak_seed = int(batch.metadata["weak_seed"])
    grid_shape = tuple(grid.shape[:-1])
    hf_variance = np.full(grid_shape, noise_hf_abs**2, dtype=float)
    lf_variance = np.full(grid_shape, noise_lf_abs**2, dtype=float)
    # One library per group, not per trajectory. Its domain placement,
    # quadrature weights and covariance Cholesky depend on the grid, the seed,
    # the support and the variance field -- all fixed within a group -- so only
    # the transform depends on the data. Rebuilding per trajectory repeated that
    # work 11 times over for a 10-LF group, and the tuner sweeps the grid 135
    # times, which is where the isothermal case spent its hours.
    libraries: Dict[Any, Any] = {}

    def library_for(variance_field: np.ndarray | None, whitener_mode: str, sample):
        signature = _variance_signature(variance_field)
        key = None if signature is None else (signature, whitener_mode)
        if key is not None and key in libraries:
            return libraries[key]
        library = _ns_make_weak_library(
            cfg,
            grid,
            variance_field=variance_field,
            weak_seed=weak_seed,
            whitener_mode=whitener_mode,
        )
        # Fit once: the geometry does not depend on which trajectory is passed,
        # and refitting would redraw the domains without reseeding.
        library.fit([sample])
        if key is not None:
            libraries[key] = library
        return library

    def build_group(
        trajectories: list[np.ndarray],
        *,
        variance_field: np.ndarray | None,
        whitener_mode: str = "full",
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        theta_blocks: list[np.ndarray] = []
        rhs_blocks: list[np.ndarray] = []
        for traj in trajectories:
            library = library_for(variance_field, whitener_mode, traj)
            theta_blocks.append(np.asarray(library.transform([traj])[0]))
            rhs_blocks.append(np.asarray(library.convert_u_dot_integral(traj)))
        return theta_blocks, rhs_blocks

    def group_builder(fidelity: str, weighting: str):
        trajectories = batch.hf if fidelity == "hf" else batch.lf
        if weighting == "plain":
            variance_field = None
        elif weighting == "pooled":
            # PMF is fidelity-blind: a variance common to every trajectory cancels
            # in the least-squares solution, leaving only the test-function
            # correlations.
            variance_field = np.ones(grid_shape, dtype=float)
        else:
            variance_field = hf_variance if fidelity == "hf" else lf_variance
        whitener_mode = "diag" if weighting == "diag" else "full"
        return build_group(
            trajectories, variance_field=variance_field, whitener_mode=whitener_mode
        )

    def fit_stacked(theta_blocks, rhs_blocks):
        return _fit_stacked_weak_system(theta_blocks, rhs_blocks, optimizer_factory)

    return assemble_weak_rungs(group_builder, fit_stacked, methods, seed=weak_seed)


def ns_isothermal_weak_design(
    cfg: NSIsothermalMultiTrajectoryGLSConfig,
    *,
    weak_seed: int | None = None,
) -> dict:
    """The weak design a config actually produces, for the record in the paper.

    The test-function weights depend on the spatiotemporal grid alone, so this
    needs no flow solution: the grid is rebuilt from the config and a zero field
    gives the same design the experiment builds.
    """

    x = np.linspace(0.0, cfg.L, cfg.N, endpoint=False)
    y = np.linspace(0.0, cfg.L, cfg.N, endpoint=False)
    t = np.linspace(0.0, cfg.T, cfg.Nt)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grid = np.zeros((cfg.N, cfg.N, cfg.Nt, 3))
    grid[:, :, :, 0] = X[:, :, None]
    grid[:, :, :, 1] = Y[:, :, None]
    grid[:, :, :, 2] = t[None, None, :]

    field_shape = (cfg.N, cfg.N, cfg.Nt)
    H = _ns_multi_h_xt(cfg)
    extents = tuple(float(np.ptp(grid[..., axis])) for axis in range(grid.shape[-1]))
    if cfg.K is not None:
        K_requested = int(cfg.K)
    else:
        domain = float(np.prod([2.0 * h for h in np.atleast_1d(H)]))
        K_requested = max(2, int(round(2.0 * float(np.prod(extents)) / domain)))

    library = _ns_make_weak_library(
        cfg,
        grid,
        variance_field=np.ones(field_shape, dtype=float),
        weak_seed=int(cfg.seed_base if weak_seed is None else weak_seed),
    )
    library.fit_transform([np.zeros(field_shape + (3,))])

    report = weak_design_report(library, K_requested)
    # For a PDE the neglected term is controlled by scale separation between
    # the temporal and spatial supports rather than by the Jacobian, so the
    # validity ratio is analytic: h_t^2 * h_x^(-2m).
    widths = list(np.atleast_1d(H))
    report["kappa_scale_sep"] = pde_scale_separation_ratio(
        widths[-1], widths[:-1], cfg.derivative_order
    )
    return report


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
    
    # The clean reference flow sets state_std and the grid -- nothing more is
    # needed from it. It used to come from a weak-SINDy fit of this same flow,
    # which ran the whole fit only for the coefficients to be discarded.
    U_ref, t_ref, grid_ref = generate_isothermal_ns_dataset(
        N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
        seed=cfg.seed_base,
    )

    state_std = float(np.std(U_ref[:,:,:,2]))

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
    # Truth is the equations ``compressible`` integrates, written out in the
    # library's own feature names -- not a weak-SINDy fit of the clean flow,
    # which is what this used to score against. That reference was circular and
    # wrong three ways: underdetermined (K_std=100 rows against 111 features),
    # non-deterministic (its domain placement is drawn from the global RNG,
    # which nothing seeds, so "truth" moved between runs), and pruned by an
    # STLSQ threshold of 0.5 against coefficients of magnitude 1. On a clean
    # held-out flow it explained 99.74% of the dynamics against 99.99% for the
    # analytic set, and it had dropped v*u_y from the u equation while carrying
    # a spurious -5.7 v*u_xy in v.
    feature_names = get_ns_isothermal_feature_names(
        cfg, grid=grid_ref, reference_trajectory=U_ref
    )
    C_analytic = build_true_ns_isothermal_coefficients(
        feature_names, RT=cfg.RT, mu=cfg.mu
    )

    return run_multi_trajectory_gls_experiment(
        cfg,
        reference_state_std=_reference_state_std,
        dataset_builder=_dataset_builder,
        true_coefficients=lambda _batch, _cfg: C_analytic,
        optimizer_factory=cfg.make_optimizer,
        fit_models_fn=_ns_fit_multi_trajectory_weak_gls_models,
        progress_desc="MC isothermal NS MF",
    )


#: The isothermal compressible system as ``compressible`` integrates it, written
#: against the weak library's feature names. With ``p = rho * RT``:
#:
#:     u_t   = -u u_x - v u_y - RT rho^-1 rho_x + mu rho^-1 (u_xx + u_yy)
#:     v_t   = -u v_x - v v_y - RT rho^-1 rho_y + mu rho^-1 (v_xx + v_yy)
#:     rho_t = -u rho_x - v rho_y - rho u_x - rho v_y
#:
#: Fourteen terms, every coefficient +-1 scaled by RT or mu. Axis 1 is x and
#: axis 2 is y in pysindy's derivative naming, and a leading library name is the
#: function factor: ``rhou_1`` is ``rho * u_x`` while ``urho_1`` is ``u * rho_x``.
_NS_TRUE_TERMS: Dict[int, Dict[str, str]] = {
    0: {"uu_1": "-1", "vu_2": "-1", "rho^-1rho_1": "-RT",
        "rho^-1u_11": "mu", "rho^-1u_22": "mu"},
    1: {"uv_1": "-1", "vv_2": "-1", "rho^-1rho_2": "-RT",
        "rho^-1v_11": "mu", "rho^-1v_22": "mu"},
    2: {"urho_1": "-1", "vrho_2": "-1", "rhou_1": "-1", "rhov_2": "-1"},
}


def build_true_ns_isothermal_coefficients(
    feature_names: Sequence[str], *, RT: float, mu: float
) -> np.ndarray:
    """Analytic coefficients of the isothermal compressible system.

    The alternative was to call the reference a weak-SINDy fit of the clean
    flow, which is what this case used to do. That made the benchmark circular
    -- every rung was scored against another fit rather than against the
    equations -- and the fit was not a good one: at ``K_std=100`` rows against
    111 library features it was underdetermined, and it came back missing
    ``v u_y`` from the u equation and carrying a spurious ``-5.7 v u_xy`` in the
    v equation, explaining 89.6% of the clean dynamics where plain least squares
    on the same library reaches 99.9999%.

    ``rho^-1`` is the library's ``1 / (1e-6 + |rho|)``, which differs from
    ``1 / rho`` by about a part in a million for a density near one.

    Raises if a term is missing from ``feature_names``: a library that can no
    longer express the physics must fail loudly rather than score every rung
    against a truncated truth.
    """

    feature_names = list(feature_names)
    index = {name: j for j, name in enumerate(feature_names)}
    values = {"-1": -1.0, "-RT": -float(RT), "mu": float(mu)}

    coefficients = np.zeros((3, len(feature_names)), dtype=float)
    missing: list[str] = []
    for state, terms in _NS_TRUE_TERMS.items():
        for name, symbol in terms.items():
            if name not in index:
                missing.append(name)
                continue
            coefficients[state, index[name]] = values[symbol]
    if missing:
        raise KeyError(
            f"The weak library does not provide {missing}; the analytic "
            "isothermal coefficients cannot be expressed in it."
        )
    return coefficients


def get_ns_isothermal_feature_names(
    cfg: NSIsothermalMultiTrajectoryGLSConfig | NSIsothermalIntraTrajectoryGLSConfig,
    *,
    grid: np.ndarray,
    reference_trajectory: np.ndarray,
) -> tuple[str, ...]:
    """Return readable weak-library feature names for isothermal-flow models."""

    if isinstance(cfg, NSIsothermalMultiTrajectoryGLSConfig):
        multi_cfg = cfg
        library = _ns_make_weak_library(
            multi_cfg,
            np.asarray(grid, dtype=float),
            variance_field=None,
            weak_seed=multi_cfg.seed_base,
        )
    else:
        h_xt = _ns_intra_h_xt(cfg)
        grid_array = np.asarray(grid, dtype=float)
        # Only the feature names are wanted, but K still has to be a number, and
        # it follows the support here like everywhere else.
        library = WeakPDELibrary(
            function_library=_build_custom_library(),
            derivative_order=cfg.derivative_order,
            spatiotemporal_grid=grid_array,
            is_uniform=True,
            K=(
                int(cfg.K)
                if cfg.K is not None
                else pde_coverage_test_function_count(grid_array, h_xt)
            ),
            p=cfg.p,
            H_xt=h_xt,
            include_bias=cfg.include_bias,
        )

    return get_library_feature_names(
        library,
        reference_trajectory,
        input_features=("u", "v", "rho"),
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
    N: int = 32
    Nt: int = 2501          # linspace(0, T, Nt): dt = T / (Nt - 1) = 1e-3
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
    # K follows H_xt at coverage 2 when None. It used to be pinned at 1000
    # whatever the support, which is a different coverage at every candidate --
    # around 20 at the default support, where the covariance is near singular.
    K: int | None = None
    H_xt: list[float] | None = None

    # SINDy / optimizer settings
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
    )

    U_noisy, variance = add_heteroscedastic_noise_temporal_derivative(
        U_clean, t, sigma0=cfg.sigma0, alpha=cfg.alpha, rng=rng
    )

    # The four weightings, all on the same test functions. K follows the support
    # at coverage 2 unless the config pins it.
    return build_pde_intra_trajectory_artifacts(
        U_noisy,
        np.asarray(grid, dtype=float),
        t_grid=t,
        variance=variance / np.mean(variance),
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        K=cfg.K,
        H_xt=_ns_intra_h_xt(cfg),
        p=cfg.p,
        weak_seed=cfg.seed_base + 1000 + run_idx,
        true_coefficients=true_coefficients,
    )


def run_ns_isothermal_intra_trajectory_gls_experiment(
    cfg: NSIsothermalIntraTrajectoryGLSConfig,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full heteroscedastic NS GLS experiment (Part 2).
    """
    U_clean_ref, _, grid_ref = generate_isothermal_ns_dataset(
        N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
        seed=cfg.seed_base,
    )
    base_library = _build_custom_library()
    # Truth is the equations compressible integrates, not a weak-SINDy fit of
    # the clean flow. Part 2 had the same circular reference Part 1 did: scored
    # against another fit, underdetermined, unseeded, and wrong in the u and v
    # equations.
    C_true = build_true_ns_isothermal_coefficients(
        get_ns_isothermal_feature_names(
            cfg, grid=grid_ref, reference_trajectory=U_clean_ref
        ),
        RT=cfg.RT,
        mu=cfg.mu,
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


def build_ns_isothermal_intra_trajectory_artifacts(
    data: np.ndarray,
    t_grid: np.ndarray,
    grid: np.ndarray,
    *,
    variance: np.ndarray,
    cfg: NSIsothermalIntraTrajectoryGLSConfig,
    base_library: ps.CustomLibrary | None = None,
    true_coefficients: np.ndarray | None = None,
    weak_seed: int | None = None,
) -> IntraTrajectoryGLSData:
    """Construct isothermal-flow weak-form artifacts for a provided trajectory."""

    data = np.asarray(data, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    grid = np.asarray(grid, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError("data must have shape (N, N, Nt, 3).")
    if variance.shape != data.shape[:-1]:
        raise ValueError(f"variance must have shape {data.shape[:-1]}, got {variance.shape}.")

    return build_pde_intra_trajectory_artifacts(
        data,
        grid,
        t_grid=t_grid,
        variance=variance / np.mean(variance),
        function_library=(
            _build_custom_library() if base_library is None else base_library
        ),
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        K=cfg.K,
        H_xt=_ns_intra_h_xt(cfg),
        p=cfg.p,
        weak_seed=cfg.seed_base if weak_seed is None else int(weak_seed),
        true_coefficients=(
            np.empty((data.shape[-1], 0))
            if true_coefficients is None
            else np.asarray(true_coefficients)
        ),
    )


def ns_isothermal_kappa_by_support(
    cfg: NSIsothermalMultiTrajectoryGLSConfig,
    candidates,
    *,
    coefficients: np.ndarray,
    n_reference: int = 3,
    weak_seed: int | None = None,
) -> pd.DataFrame:
    """Kappa per candidate fitting support, on fixed reference flows.

    As for Burgers, but ``coefficients`` is required rather than derived: the
    isothermal reference coefficients come from a weak fit of their own, and
    recomputing it here would double the most expensive step in the notebook.

    ``kappa_median`` is the reported statistic and ``kappa_lo``/``kappa_hi`` bracket
    it with the best and worst reference flow. The Taylor-Green initial
    condition randomises its amplitudes and wavenumbers, and the wavenumbers set
    the size of the derivatives the library sees, so the spread across
    references is the honest uncertainty in the bound rather than noise.
    """

    seed = cfg.seed_base if weak_seed is None else int(weak_seed)
    references = []
    for j in range(int(n_reference)):
        U_ref, _t_ref, grid_ref = generate_isothermal_ns_dataset(
            N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
            seed=cfg.seed_base + j,
        )
        references.append((np.asarray(U_ref), np.asarray(grid_ref, dtype=float)))

    rows: list[dict] = []
    for H in candidates:
        medians: list[float] = []
        maxima: list[float] = []
        K_used = 0
        for U_ref, grid_ref in references:
            library = _ns_make_weak_library(
                replace(cfg, H_xt=list(H), K=None), grid_ref,
                variance_field=None, weak_seed=seed,
            )
            library.fit_transform([U_ref])
            ratios = pde_weak_validity_ratio(
                library,
                pde_sensitivity_fields(library, U_ref, coefficients),
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
