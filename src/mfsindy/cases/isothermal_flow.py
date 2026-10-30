# %% ns_isothermal_utils.py
"""
Isothermal compressible Navier–Stokes utilities.

Includes:

1) PDE + dataset generator
2) Heteroscedastic noise sigma_i = a s_i + b |u_i| per component (Part 2, as in the ODE cases)
3) Coefficient error metrics + generic MC wrapper
4) Multi-fidelity SINDy experiment (Part 1: HF / LF / MF / MF_w)
5) Heteroscedastic GLS experiment (Part 2: No weighting / Variance GLS / Ones GLS)
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass, replace
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import pysindy as ps
from pysindy.utils import AxesArray

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
    refine: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single trajectory of isothermal compressible flow, on an N x N grid.

    The flow is solved on a grid ``refine`` times finer and sampled onto the
    N x N one. Solving on the measurement grid itself makes the data a solution
    of the finite-difference scheme rather than of the PDE, and the analytic
    coefficients then leave a residual on clean data: 2-8% at N = 32, 0.6-1.9%
    at N = 64, at or above the 1% HF noise, which the HF-trusting models fit with
    spurious terms. Solved at 4 x 32 = 128 the residual is 0.2-0.7%.

    Returns
    -------
    U_clean : (N, N, Nt, 3)
        States (u, v, rho). Read-only: it may be shared through the cache.
    t : (Nt,)
        Time vector.
    grid : (N, N, Nt, 3)
        Spatiotemporal grid (x, y, t).
    """

    U = _solve_isothermal_ns(int(N), int(Nt), float(L), float(T), float(mu), float(RT), int(seed), int(refine))
    t = np.linspace(0.0, T, Nt)
    x = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    grid = np.zeros((N, N, Nt, 3))
    grid[:, :, :, 0] = X[:, :, None]
    grid[:, :, :, 1] = Y[:, :, None]
    grid[:, :, :, 2] = t[None, None, :]
    return U, t, grid


@functools.lru_cache(maxsize=16)
def _solve_isothermal_ns(N, Nt, L, T, mu, RT, seed, refine) -> np.ndarray:
    """The flow of ``generate_isothermal_ns_dataset``, cached: a Part II run is
    rebuilt once per candidate support, and the fine solve is the slow part."""

    rng = np.random.default_rng(seed)
    n_fine = N * refine
    x = np.linspace(0.0, L, n_fine, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    U0, V0, RHO0 = make_initial_condition(X, Y, L, rng=rng)
    y0 = np.stack([U0, V0, RHO0], axis=-1)

    sol = solve_ivp(
        compressible,
        (0.0, T),
        y0.reshape(-1),
        t_eval=np.linspace(0.0, T, Nt),
        args=(x[1] - x[0], n_fine, mu, RT),
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    U = sol.y.reshape(n_fine, n_fine, 3, -1).transpose(0, 1, 3, 2)[::refine, ::refine]
    U = np.ascontiguousarray(U)
    U.flags.writeable = False
    return U


# ---------------------------------------------------------------------------
# Conservative-form library + reference coefficients (Parts 1 and 2)
# ---------------------------------------------------------------------------
#
# The momentum equations are identified for rho*u and rho*v rather than for u
# and v. In primitive form every pressure and viscous term carries 1/rho, and
# 1/rho evaluated on noisy data is biased and heavy-tailed (errors in
# variables): at 25% noise the fits shrank every rho^-1 term to 0.4-0.8 of its
# value and spurious terms made up the rest. In conservative form the system is
#
#     (rho u)_t = -(rho u u)_x - (rho u v)_y - RT rho_x + mu (u_xx + u_yy)
#     (rho v)_t = -(rho u v)_x - (rho v v)_y - RT rho_y + mu (v_xx + v_yy)
#     rho_t     = -(rho u)_x - (rho v)_y
#
# which is what ``compressible`` integrates, rewritten exactly. Every term is a
# derivative of a polynomial in (u, v, rho), so the weak form moves the whole
# derivative onto the test function: the library is the integrals of every
# monomial of degree <= 3 against phi and its derivatives, and products of
# independent noise are unbiased.

#: The monomials of (rho, u, v) of degree 1 to 3, as the library's input fields.
_NS_MONOMIALS: tuple[tuple[int, ...], ...] = tuple(
    combo
    for degree in (1, 2, 3)
    for combo in itertools.combinations_with_replacement((2, 0, 1), degree)
)
_NS_STATE_NAMES = {0: "u", 1: "v", 2: "rho"}
_NS_MONOMIAL_NAMES: tuple[str, ...] = tuple(
    "*".join(_NS_STATE_NAMES[i] for i in combo) for combo in _NS_MONOMIALS
)
#: The fields whose time derivatives are identified: rho*u, rho*v and rho.
_NS_TARGETS: tuple[int, ...] = tuple(
    _NS_MONOMIAL_NAMES.index(name) for name in ("rho*u", "rho*v", "rho")
)


def ns_monomial_fields(U: np.ndarray) -> np.ndarray:
    """The monomials of ``(u, v, rho)`` as fields, shape ``(..., 19)``."""

    U = np.asarray(U, dtype=float)
    return np.stack([np.prod(U[..., list(combo)], axis=-1) for combo in _NS_MONOMIALS], axis=-1)


def ns_target_variance(U_clean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Noise variance of the targets ``(rho u, rho v, rho)`` from that of ``(u, v, rho)``.

    For independent noise e on each component, rho*u + noise has variance
    rho^2 var(e_u) + u^2 var(e_rho) + var(e_u) var(e_rho), exactly.
    """

    u, v, rho = (U_clean[..., i] for i in range(3))
    var_u, var_v, var_rho = (variance[..., i] for i in range(3))
    return np.stack(
        [
            rho**2 * var_u + u**2 * var_rho + var_u * var_rho,
            rho**2 * var_v + v**2 * var_rho + var_v * var_rho,
            var_rho,
        ],
        axis=-1,
    )


class ConservativeWeakPDELibrary(DedupedWeakPDELibrary):
    """A weak library on the monomials of ``(u, v, rho)``, identifying the
    conservative targets.

    It takes the flow as ``(u, v, rho)`` like any other library and expands it
    to the monomial fields itself, so callers and stored data stay three
    components wide; ``convert_u_dot_integral`` keeps only the targets. Use
    ``include_interaction=False``: the pure integral terms already hold every
    derivative of every monomial.
    """

    def fit(self, x_full, y=None):
        return super().fit([ns_monomial_fields(x) for x in x_full], y)

    def transform(self, x_full):
        out = super().transform([ns_monomial_fields(x) for x in x_full])
        # pysindy keeps the data and its derivatives on every subdomain, rebuilt
        # on each transform and read by nothing after it: 2 GB per library at
        # N = 64, with forty libraries alive in a Part II run.
        self.x_k = self.dx_k_j = self.dfx_k_j = None
        return out

    def convert_u_dot_integral(self, u):
        full = np.asarray(super().convert_u_dot_integral(ns_monomial_fields(u)))
        return AxesArray(full[:, list(_NS_TARGETS)], {"ax_sample": 0, "ax_coord": 1})


class WeightedConservativeWeakPDELibrary(WeightedWeakPDELibrary, ConservativeWeakPDELibrary):
    """The weighted library on the conservative targets.

    The method order puts the expansion between the whitening and pysindy, so
    the whitener sees the three target columns: a per-component variance is
    given for (rho u, rho v, rho), see :func:`ns_target_variance`.
    """


def _build_custom_library():
    """The function library on the monomial fields: the fields themselves."""

    return ps.PolynomialLibrary(degree=1, include_bias=False)


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
    N: int = 32
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
    n_ensemble_models: int = 100  # as every case; the bags are most of a run's time here
    # About 3 GB per worker, so two fit in 16 GB.
    n_jobs: int = 2

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
        include_interaction=False,
    )
    extents = tuple(float(np.ptp(grid[..., axis])) for axis in range(grid.shape[-1]))
    H = _ns_multi_h_xt(cfg)
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
        # Genuinely unweighted, as in the other cases. Passing a field of ones
        # here would still apply the weak-SINDy whitening, which made the HF, LF
        # and MF rungs weighted and identical to PMF.
        return ConservativeWeakPDELibrary(deduplicate=cfg.deduplicate, **common_kwargs)
    return WeightedConservativeWeakPDELibrary(
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
    # One variance per fidelity group, as in the ODE cases. The noise on rho u is
    # sigma^2 (rho^2 + u^2 + sigma^2), not constant, but the group weighting
    # only needs the ratio between groups, and that is sigma_hf^2 / sigma_lf^2
    # up to the sigma^2 term.
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
        K_requested = max(2, int(round(cfg.coverage * float(np.prod(extents)) / domain)))

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


#: The isothermal compressible system in conservative form (see the library
#: section), against the weak library's feature names: a monomial name with
#: pysindy's derivative suffix, axis 1 being x and axis 2 y, so ``rho*u*u_1`` is
#: ``(rho u u)_x``. Twelve terms, every coefficient +-1 scaled by RT or mu.
_NS_TRUE_TERMS: Dict[int, Dict[str, str]] = {
    0: {"rho*u*u_1": "-1", "rho*u*v_2": "-1", "rho_1": "-RT", "u_11": "mu", "u_22": "mu"},
    1: {"rho*u*v_1": "-1", "rho*v*v_2": "-1", "rho_2": "-RT", "v_11": "mu", "v_22": "mu"},
    2: {"rho*u_1": "-1", "rho*v_2": "-1"},
}


def build_true_ns_isothermal_coefficients(
    feature_names: Sequence[str], *, RT: float, mu: float
) -> np.ndarray:
    """Analytic coefficients of the isothermal compressible system, in
    conservative form, one row per target (rho u, rho v, rho).

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
    """The weak library's feature names; they depend on neither support nor K."""

    library = ConservativeWeakPDELibrary(
        function_library=_build_custom_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=np.asarray(grid, dtype=float),
        is_uniform=True,
        include_bias=cfg.include_bias,
        include_interaction=False,
        K=2,
        p=cfg.p,
    )
    return get_library_feature_names(
        library, reference_trajectory, input_features=_NS_MONOMIAL_NAMES
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
    Nt: int = 1001          # linspace(0, T, Nt): dt = T / (Nt - 1) = 1e-3
    L: float = 5.0
    T: float = 1.0
    n_traj: int = 10        # flows per run, each from its own initial condition

    # physical parameters
    mu: float = 1.0
    RT: float = 1.0

    # per-component noise, as in every Part II case (see ns_noise_variance):
    # sigma_i(x, y, t) = noise_floor * s_i + noise_level * |u_i(x, y, t)|
    noise_level: float = 0.15
    noise_floor: float = 1e-4

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
    n_ensemble_models: int = 100  # as every case; the bags are most of a run's time here
    # A run holds 10 flows x 4 libraries on one shared set of test functions
    # (_share_test_functions): under 2 GB at N = 32, 7.6 GB at N = 64.
    n_jobs: int = 1

    results_filename: str = "ns_isothermal_weighted_errors.csv"


def ns_noise_variance(U_clean: np.ndarray, cfg) -> np.ndarray:
    """Pointwise noise variance per state component, shape of ``U_clean``.

    sigma_i = noise_floor * s_i + noise_level * |u_i|: the two-component
    measurement-error model of the ODE cases (lorenz_noise_variance), applied to
    u, v and rho as if each had its own sensor. s_i is the spread of component i
    over the flow, so the floor is in that component's units.
    """

    U_clean = np.asarray(U_clean, dtype=float)
    scale = np.std(U_clean, axis=(0, 1, 2))
    sigma = cfg.noise_floor * scale + cfg.noise_level * np.abs(U_clean)
    return np.maximum(sigma ** 2, 1e-12)


def ns_run_trajectories(cfg: NSIsothermalIntraTrajectoryGLSConfig, rng: np.random.Generator):
    """The cfg.n_traj flows of one Part II run, as ``(t, grid, clean, noisy, variance)``.

    Flows are ``(N, N, Nt, 3)`` and so are their variances. Each flow starts
    from its own randomised Taylor-Green condition and gets its own noise, drawn
    from ns_noise_variance. The Monte Carlo builder, main2.py and the notebook
    all draw through this function, so they show the same data.
    """

    clean, noisy, variances = [], [], []
    for _ in range(cfg.n_traj):
        U_clean, t, grid = generate_isothermal_ns_dataset(
            N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
            seed=int(rng.integers(2**31)),
        )
        variance = ns_noise_variance(U_clean, cfg)
        clean.append(U_clean)
        noisy.append(U_clean + np.sqrt(variance) * rng.standard_normal(size=U_clean.shape))
        variances.append(variance)
    return t, np.asarray(grid, dtype=float), clean, noisy, variances


#: What pysindy builds per test function at construction: the supports and the
#: weight tensors on them, 1.7 GB per library at N = 64.
_TEST_FUNCTION_ATTRS = ("inds_k", "fulltweights", "fullweights0", "fullweights1")


def _share_test_functions(libraries, template) -> None:
    """Point every library at ``template``'s test functions.

    Libraries built on one seed, grid and support have identical test functions
    but each keeps its own copy, forty of them in a Part II run. Their whiteners
    are already built, so sharing changes nothing but the memory; the supports
    are checked first so that it cannot replace different test functions.
    """

    for library in libraries:
        if library is template:
            continue
        same = len(library.inds_k) == len(template.inds_k) and all(
            np.array_equal(a, b)
            for ka, kb in zip(library.inds_k, template.inds_k)
            for a, b in zip(ka, kb)
        )
        if not same:
            raise ValueError("libraries to share test functions must be built on the same ones.")
        for name in _TEST_FUNCTION_ATTRS:
            setattr(library, name, getattr(template, name))


def build_ns_artifacts(
    data: np.ndarray,
    grid: np.ndarray,
    t: np.ndarray,
    target_variance: np.ndarray,
    cfg: NSIsothermalIntraTrajectoryGLSConfig,
    *,
    weak_seed: int,
    true_coefficients: np.ndarray,
    H_xt=None,
    template=None,
) -> IntraTrajectoryGLSData:
    """The four Part II weightings for one flow, on the conservative library.

    As :func:`mfsindy.experiments.build_pde_intra_trajectory_artifacts`, with
    the conservative libraries in place of the plain ones: every rung on the
    same test functions (one seed), K following the support at coverage 2
    unless pinned. ``target_variance`` is per target, ``(..., 3)`` for
    (rho u, rho v, rho), so each equation is whitened by its own noise.

    The rungs share one copy of the test-function weights, and so do the flows
    of a run when ``template`` (a library built on the same seed) is given.
    """

    H = _ns_intra_h_xt(cfg) if H_xt is None else list(H_xt)
    common = dict(
        function_library=_build_custom_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        is_uniform=True,
        include_bias=cfg.include_bias,
        include_interaction=False,
        H_xt=H,
        K=int(cfg.K) if cfg.K is not None else pde_coverage_test_function_count(grid, H, cfg.coverage),
        p=cfg.p,
    )

    def seeded(cls, **kwargs):
        np.random.seed(int(weak_seed))
        return cls(**common, **kwargs)

    libraries = {
        "No weighting": seeded(ConservativeWeakPDELibrary),
        "Variance GLS": seeded(WeightedConservativeWeakPDELibrary, spatiotemporal_weights=target_variance),
        "Ones GLS": seeded(WeightedConservativeWeakPDELibrary, spatiotemporal_weights=np.ones(grid.shape[:-1])),
        "Variance diag": seeded(
            WeightedConservativeWeakPDELibrary, spatiotemporal_weights=target_variance, whitener_mode="diag"
        ),
    }
    _share_test_functions(libraries.values(), template or libraries["No weighting"])
    return IntraTrajectoryGLSData(
        data=data, t_argument=t, libraries=libraries, true_coefficients=np.asarray(true_coefficients)
    )


def _make_ns_gls_artifacts(
    run_idx: int,
    cfg: NSIsothermalIntraTrajectoryGLSConfig,
    rng: np.random.Generator,
    *,
    true_coefficients: np.ndarray,
) -> list[IntraTrajectoryGLSData]:
    """The four weightings for each flow of one NS GLS run.

    Every flow and every rung is placed on the same test functions (one seed),
    so the rungs differ by covariance model alone. The variances are passed as
    they are, not rescaled per flow: the flows' weak systems are stacked, and a
    per-flow rescaling would change their relative weights.
    """

    t, grid, clean, noisy, variances = ns_run_trajectories(cfg, rng)
    out, template = [], None
    for data, U_clean, variance in zip(noisy, clean, variances):
        out.append(build_ns_artifacts(
            data, grid, t, ns_target_variance(U_clean, variance), cfg,
            weak_seed=cfg.seed_base + 1000 + run_idx,
            true_coefficients=true_coefficients, template=template,
        ))
        template = out[0].libraries["No weighting"]
    return out


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

    def builder(run_idx: int, cfg: NSIsothermalIntraTrajectoryGLSConfig) -> list[IntraTrajectoryGLSData]:
        # Seeded by run, not drawn from one stream: the runner builds a run once
        # per candidate support, and every support must see the same data.
        return _make_ns_gls_artifacts(
            run_idx,
            cfg,
            np.random.default_rng(cfg.seed_base + 100 * run_idx),
            true_coefficients=C_true,
        )

    return run_intra_trajectory_gls_experiment(
        cfg,
        run_builder=builder,
        progress_desc="MC isothermal NS GLS",
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
            # The library's inputs are the monomial fields, so the sensitivity is
            # taken with respect to them: each monomial is treated as a state of
            # its own, which makes kappa a rougher diagnostic here than for the
            # cases whose library sees the state directly.
            ratios = pde_weak_validity_ratio(
                library,
                pde_sensitivity_fields(library, ns_monomial_fields(U_ref), coefficients),
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
