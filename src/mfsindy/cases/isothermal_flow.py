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

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

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
    assemble_weak_rungs,
    WeakValidationBlock,
    get_library_feature_names,
    run_intra_trajectory_gls_experiment,
    run_multi_trajectory_gls_experiment,
)
from mfsindy.weighted_weak_pde_library import (
    DedupedWeakPDELibrary,
    WeightedWeakPDELibrary,
    usable_test_functions,
    weak_design_report,
)


_NS_PART1_TUNED_HYPERPARAMS_FILENAME = "navierstokes_part1_tuned_hyperparams.json"
_NS_PART1_MODELS = ("HF", "LF", "MF", "MF_w")


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_ns_part1_tuned_hyperparams(
    params_by_model: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(params_by_model, dict):
        raise ValueError("Expected a mapping of model names to hyperparameters.")

    missing_models = [
        model_name for model_name in _NS_PART1_MODELS if model_name not in params_by_model
    ]
    if missing_models:
        missing = ", ".join(missing_models)
        raise ValueError(f"Missing tuned hyperparameters for: {missing}.")

    normalized: dict[str, dict[str, Any]] = {}
    for model_name in _NS_PART1_MODELS:
        params = params_by_model[model_name]
        if not isinstance(params, dict):
            raise ValueError(f"Expected a mapping of hyperparameters for {model_name!r}.")

        if "stlsq_threshold" not in params:
            raise ValueError(f"Missing 'stlsq_threshold' for {model_name!r}.")
        if "H_xt" not in params:
            raise ValueError(f"Missing 'H_xt' for {model_name!r}.")

        h_xt = [float(value) for value in params["H_xt"]]
        if len(h_xt) != 3:
            raise ValueError(
                f"Expected 'H_xt' to contain three entries for {model_name!r}, got {len(h_xt)}."
            )

        normalized[model_name] = {
            "stlsq_threshold": float(params["stlsq_threshold"]),
            "H_xt": h_xt,
        }

    return normalized


def _ns_part1_tuned_hyperparams_candidates(
    *,
    results_dir: str | os.PathLike[str] | None = None,
    filename: str = _NS_PART1_TUNED_HYPERPARAMS_FILENAME,
) -> list[Path]:
    candidates: list[Path] = []
    if results_dir is not None:
        candidates.append(Path(results_dir) / filename)

    repo_root = _repo_root()
    candidates.extend(
        [
            repo_root / "examples" / "isothermal_flow" / "results" / filename,
            repo_root / "results" / filename,
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        key = candidate.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def load_ns_part1_tuned_hyperparams(
    *,
    results_dir: str | os.PathLike[str] | None = None,
    filename: str = _NS_PART1_TUNED_HYPERPARAMS_FILENAME,
) -> dict[str, dict[str, Any]] | None:
    """Load cached Part 1 Navier-Stokes tuning results when available."""

    for path in _ns_part1_tuned_hyperparams_candidates(results_dir=results_dir, filename=filename):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            return _normalize_ns_part1_tuned_hyperparams(json.load(handle))
    return None


def save_ns_part1_tuned_hyperparams(
    params_by_model: dict[str, Any],
    *,
    results_dir: str | os.PathLike[str] | None = None,
    filename: str = _NS_PART1_TUNED_HYPERPARAMS_FILENAME,
) -> Path:
    """Persist Part 1 Navier-Stokes tuning results for reuse."""

    normalized = _normalize_ns_part1_tuned_hyperparams(params_by_model)
    if results_dir is None:
        target = _repo_root() / "examples" / "isothermal_flow" / "results" / filename
    else:
        target = Path(results_dir) / filename

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
        handle.write("\n")
    return target


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
    H_xt: list[float] | tuple[float, float, float] | None = None,
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
    
    h_xt = list(H_xt) if H_xt is not None else [L / 10.0, L / 10.0, T / 10.0]

    weak_lib_ref = WeightedWeakPDELibrary(
        function_library=_build_custom_library(),  # separate instance is fine
        derivative_order=derivative_order,
        spatiotemporal_grid=grid,
        K=K_ref,
        p=p,
        spatiotemporal_weights=np.ones((U_clean.shape[0],U_clean.shape[1],U_clean.shape[2])),
        H_xt=h_xt,
        include_bias=include_bias,
    )

    opt_ref = ps.STLSQ(threshold=0.5, alpha=1e-12)
    model_ref = ps.SINDy(feature_library=weak_lib_ref, optimizer=opt_ref)

    model_ref.fit(U_clean, t=t)

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
    n_lf: int = 10
    n_hf: int = 1

    # relative noise levels w.r.t. std(U_clean_ref)
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    # grid / time
    N: int = 32
    Nt: int = 100
    Nt_std: int = 500
    L: float = 5.0
    T: float = 0.1
    T_std : float = 0.5

    # physical parameters
    mu: float = 1.0
    RT: float = 1.0

    # weak-library settings
    derivative_order: int = 2
    include_bias: bool = False
    p: int = 2
    K: int | None = None       # derived from H_xt when None
    clamp: bool = True         # cap K at the usable rank and drop duplicate supports
    K_std: int = 100
    H_xt: list[float] | None = None

    # SINDy / optimizer settings
    stlsq_threshold: float = 0.5
    stlsq_alpha: float = 1e-12
    n_ensemble_models: int = 100

    # randomness
    seed_base: int = 0

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
        metadata={
            "grid": grid,
            "weak_seed": cfg.seed_base + 100_000 + run_idx,
        },
    )


def _ns_library(batch: MultiTrajectoryGLSData, cfg: NSIsothermalMultiTrajectoryGLSConfig):
    return WeakPDELibrary(
        function_library=_build_custom_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=batch.metadata["grid"],
        is_uniform=True,
        K=cfg.K,
        p=cfg.p,
        H_xt=_ns_multi_h_xt(cfg),
        include_bias=cfg.include_bias,
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
    # K and the support width are not independent: one test function occupies
    # prod(2H) of the grid, so tiling it once takes that ratio many of them.
    common_kwargs = dict(
        function_library=_build_custom_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        is_uniform=True,
        include_bias=cfg.include_bias,
    )
    # K and the support width are not independent: one test function occupies
    # prod(2H) of the grid, so tiling it once takes that ratio many of them.
    extents = tuple(float(np.ptp(grid[..., axis])) for axis in range(grid.shape[-1]))
    H = _ns_multi_h_xt(cfg)
    common_kwargs["H_xt"] = H
    if cfg.p is not None:
        common_kwargs["p"] = cfg.p
    if cfg.K is not None:
        K_requested = int(cfg.K)
    else:
        domain = float(np.prod([2.0 * h for h in np.atleast_1d(H)]))
        K_requested = max(2, int(round(float(np.prod(extents)) / domain)))

    def probe(K):
        probe_library = WeakPDELibrary(K=K, **common_kwargs)
        probe_library.fit([np.zeros(grid.shape[:-1] + (3,))])
        return probe_library

    common_kwargs["K"] = (
        usable_test_functions(probe, grid.shape[:-1] + extents + (tuple(H),), K_requested)
        if cfg.clamp
        else K_requested
    )
    if variance_field is None:
        # Genuinely unweighted, as in the other cases. Passing a field of ones
        # here would still apply the weak-SINDy whitening, which made the HF, LF
        # and MF rungs weighted and identical to PMF.
        return DedupedWeakPDELibrary(deduplicate=cfg.clamp, **common_kwargs)
    return WeightedWeakPDELibrary(
        spatiotemporal_weights=variance_field,
        whitener_mode=whitener_mode,
        deduplicate=cfg.clamp,
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
) -> np.ndarray:
    optimizer = optimizer_factory()
    theta = np.vstack(theta_blocks)
    rhs = np.vstack(rhs_blocks)
    optimizer.fit(theta, rhs)
    coef_list = getattr(optimizer, "coef_list", None)
    if coef_list:
        arr = np.asarray(coef_list)
        if arr.ndim == 3:
            return np.median(arr, axis=0)
    return np.asarray(optimizer.coef_)


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
    def build_group(
        trajectories: list[np.ndarray],
        *,
        variance_field: np.ndarray | None,
        whitener_mode: str = "full",
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        theta_blocks: list[np.ndarray] = []
        rhs_blocks: list[np.ndarray] = []
        for traj in trajectories:
            theta, rhs = _ns_build_weak_block(
                traj,
                cfg,
                grid=grid,
                variance_field=variance_field,
                weak_seed=weak_seed,
                whitener_mode=whitener_mode,
            )
            theta_blocks.append(theta)
            rhs_blocks.append(rhs)
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
        K_requested = max(2, int(round(float(np.prod(extents)) / domain)))

    library = _ns_make_weak_library(
        cfg,
        grid,
        variance_field=np.ones(field_shape, dtype=float),
        weak_seed=int(cfg.seed_base if weak_seed is None else weak_seed),
    )
    library.fit_transform([np.zeros(field_shape + (3,))])
    return weak_design_report(library, K_requested)


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
        H_xt=cfg.H_xt,
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
        H_xt=cfg.H_xt,
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
    return run_multi_trajectory_gls_experiment(
        cfg,
        reference_state_std=_reference_state_std,
        dataset_builder=_dataset_builder,
        library_builder=_ns_library,
        true_coefficients=lambda _batch, _cfg: C_true,
        optimizer_factory=cfg.make_optimizer,
        fit_models_fn=_ns_fit_multi_trajectory_weak_gls_models,
        progress_desc="MC isothermal NS MF",
    )


def fit_ns_isothermal_multi_trajectory_coefficients(
    cfg: NSIsothermalMultiTrajectoryGLSConfig,
    *,
    hf_trajectories: list[np.ndarray],
    lf_trajectories: list[np.ndarray],
    t_grid: np.ndarray,
    grid: np.ndarray,
    noise_hf_abs: float,
    noise_lf_abs: float,
    weak_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Fit HF/LF/MF/MF_w isothermal-flow coefficients on arbitrary trajectory sets."""

    batch = MultiTrajectoryGLSData(
        hf=hf_trajectories,
        lf=lf_trajectories,
        t_argument=np.asarray(t_grid, dtype=float),
        metadata={
            "grid": np.asarray(grid, dtype=float),
            "weak_seed": cfg.seed_base if weak_seed is None else int(weak_seed),
        },
    )
    return _ns_fit_multi_trajectory_weak_gls_models(
        batch,
        cfg,
        cfg.make_optimizer,
        t_argument=t_grid,
        noise_hf_abs=noise_hf_abs,
        noise_lf_abs=noise_lf_abs,
    )


def build_ns_isothermal_weak_validation_blocks(
    cfg: NSIsothermalMultiTrajectoryGLSConfig | NSIsothermalIntraTrajectoryGLSConfig,
    trajectories: list[np.ndarray],
    *,
    grid: np.ndarray,
    weak_seed: int | None = None,
    group_name: str = "validation",
) -> list[WeakValidationBlock]:
    """Build held-out weak-form validation blocks for isothermal-flow trajectories."""

    if isinstance(cfg, NSIsothermalMultiTrajectoryGLSConfig):
        multi_cfg = cfg
    else:
        multi_cfg = NSIsothermalMultiTrajectoryGLSConfig(
            n_lf=1,
            n_hf=1,
            noise_lf_rel=0.0,
            noise_hf_rel=0.0,
            N=cfg.N,
            Nt=cfg.Nt,
            Nt_std=cfg.Nt,
            L=cfg.L,
            T=cfg.T,
            T_std=cfg.T,
            mu=cfg.mu,
            RT=cfg.RT,
            derivative_order=cfg.derivative_order,
            include_bias=cfg.include_bias,
            p=cfg.p,
            K=cfg.K,
            K_std=cfg.K_ref,
            H_xt=cfg.H_xt,
            stlsq_threshold=cfg.stlsq_threshold,
            stlsq_alpha=cfg.stlsq_alpha,
            n_ensemble_models=cfg.n_ensemble_models,
            seed_base=cfg.seed_base,
        )

    grid = np.asarray(grid, dtype=float)
    weak_seed = multi_cfg.seed_base if weak_seed is None else int(weak_seed)
    blocks: list[WeakValidationBlock] = []
    for traj_idx, trajectory in enumerate(trajectories):
        theta, rhs = _ns_build_weak_block(
            trajectory,
            multi_cfg,
            grid=grid,
            variance_field=None,
            weak_seed=weak_seed,
        )
        blocks.append(
            WeakValidationBlock(
                theta=theta,
                rhs=rhs,
                group=group_name,
                trajectory=traj_idx,
                block=0,
            )
        )
    return blocks


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
        library = WeakPDELibrary(
            function_library=_build_custom_library(),
            derivative_order=cfg.derivative_order,
            spatiotemporal_grid=np.asarray(grid, dtype=float),
            is_uniform=True,
            K=cfg.K,
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
    H_xt: list[float] | None = None

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
    h_xt = _ns_intra_h_xt(cfg)

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
        H_xt=cfg.H_xt,
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

    base_library = _build_custom_library() if base_library is None else base_library
    variance_scaled = variance / np.mean(variance)
    tf_seed = cfg.seed_base if weak_seed is None else int(weak_seed)
    h_xt = _ns_intra_h_xt(cfg)

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

    return IntraTrajectoryGLSData(
        data=data,
        t_argument=t_grid,
        libraries={
            "No weighting": weak_lib,
            "Variance GLS": weighted_weak_lib_var,
            "Ones GLS": weighted_weak_lib_ones,
        },
        true_coefficients=
        np.empty((data.shape[-1], 0)) if true_coefficients is None else np.asarray(true_coefficients),
    )
