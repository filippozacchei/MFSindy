"""Two-dimensional incompressible flow in vorticity form.

The sixth benchmark, an alternative to the isothermal compressible flow: the same
observation design (one flow per run, observed through windows a quarter of the
domain), the same weak, conservative-form library, but an incompressible flow,
so no density, no shocks and a single equation to identify.

The flow is observed as three fields, the velocity (u, v) and the vorticity w,
as a PIV measurement gives them; the velocity is not a local function of the
vorticity (it comes from a Poisson solve over the whole domain), so it is
observed rather than reconstructed, and it enters the library as an input. Only
the vorticity equation is identified, in conservative form,

    w_t = -(u w)_x - (v w)_y + nu (w_xx + w_yy),

which is the advective form rewritten exactly, the velocity being
divergence-free. Every term is a derivative of a monomial in (u, v, w), so the
weak form moves the whole derivative onto the test function, as for the
isothermal flow.

Part 1: multi-fidelity identification (HF / LF / MF / ... / MF_w).
Part 2: heteroscedastic GLS (No weighting / Ones GLS / Variance diag / Variance GLS).
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

from mfsindy.cases.isothermal_flow import (
    _fit_stacked_weak_system,
    _fourier_diff,
    _share_test_functions,
    ns_observation_windows,
)
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
from mfsindy.experiments.artifacts import pde_coverage_test_function_count
from mfsindy.experiments.multi_trajectory import _variance_signature
from mfsindy.weighted_weak_pde_library import (
    DedupedWeakPDELibrary,
    WeightedWeakPDELibrary,
    pde_sensitivity_fields,
    pde_weak_validity_ratio,
)


# ---------------------------------------------------------------------------
# Flow: vorticity-streamfunction, Fourier pseudo-spectral
# ---------------------------------------------------------------------------

def _wavenumbers(N: int, L: float):
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    K2_inv = np.where(K2 > 0.0, 1.0 / np.where(K2 > 0.0, K2, 1.0), 0.0)
    return KX, KY, K2, K2_inv


def velocity_from_vorticity(w: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of a periodic vorticity field: -Laplacian(psi) = w, u = psi_y, v = -psi_x."""

    KX, KY, _, K2_inv = _wavenumbers(w.shape[0], L)
    psi_hat = np.fft.fft2(w) * K2_inv
    u = np.real(np.fft.ifft2(1j * KY * psi_hat))
    v = np.real(np.fft.ifft2(-1j * KX * psi_hat))
    return u, v


def vorticity_rhs(t, w_flat, N, L, nu):
    """w_t = -(u w_x + v w_y) + nu Laplacian(w), Fourier pseudo-spectral, for solve_ivp."""

    w = w_flat.reshape(N, N)
    dx = L / N
    u, v = velocity_from_vorticity(w, L)
    advection = u * _fourier_diff(w, dx, 0, 1) + v * _fourier_diff(w, dx, 1, 1)
    diffusion = _fourier_diff(w, dx, 0, 2) + _fourier_diff(w, dx, 1, 2)
    return (-advection + nu * diffusion).reshape(-1)


def make_initial_vorticity(X, Y, L, rng: np.random.Generator) -> np.ndarray:
    """A random vorticity field of a few large modes, scaled to a peak speed of 1.

    Three modes of different wavenumbers, so the nonlinear term does not vanish
    (a single Taylor-Green mode is a steady solution of the inviscid equations).
    Wavenumbers are 1 or 2 periods per side and the amplitudes random, as for the
    isothermal flow; phases are not randomised, the dynamics being translation
    invariant. The peak speed is set to 1 so every flow has the same velocity
    scale, and so the same advective time over the horizon.
    """

    k_x, k_y = (int(rng.integers(1, 3)) for _ in range(2))
    a = 0.5 + rng.random(3)
    kx, ky = 2.0 * np.pi * k_x / L, 2.0 * np.pi * k_y / L
    w = (a[0] * np.cos(kx * X) + a[1] * np.cos(ky * Y)
         + a[2] * np.sin(kx * X) * np.sin(ky * Y))
    u, v = velocity_from_vorticity(w, L)
    return w / np.max(np.hypot(u, v))


@functools.lru_cache(maxsize=2)
def _solve_vorticity(N, Nt, L, T, nu, seed) -> np.ndarray:
    """The flow of ``generate_vorticity_dataset``, cached: a Part II run is rebuilt
    once per candidate support, and the solve is the slow part."""

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    w0 = make_initial_vorticity(X, Y, L, rng)
    sol = solve_ivp(vorticity_rhs, (0.0, T), w0.reshape(-1), t_eval=np.linspace(0.0, T, Nt),
                    args=(N, L, nu), method="RK45", rtol=1e-8, atol=1e-8)
    W = sol.y.reshape(N, N, -1)
    U = np.empty((N, N, Nt, 3))
    for n in range(Nt):
        U[:, :, n, 0], U[:, :, n, 1] = velocity_from_vorticity(W[:, :, n], L)
    U[:, :, :, 2] = W
    U.flags.writeable = False
    return U


def generate_vorticity_dataset(N: int = 128, Nt: int = 1001, L: float = 5.0, T: float = 1.0,
                               nu: float = 0.05, seed: int = 1):
    """One flow on an N x N periodic grid: ``(U, t, grid)``, U of shape
    (N, N, Nt, 3) holding (u, v, w), grid (N, N, Nt, 3) holding (x, y, t)."""

    U = _solve_vorticity(int(N), int(Nt), float(L), float(T), float(nu), int(seed))
    t = np.linspace(0.0, T, Nt)
    x = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    grid = np.zeros((N, N, Nt, 3))
    grid[:, :, :, 0] = X[:, :, None]
    grid[:, :, :, 1] = Y[:, :, None]
    grid[:, :, :, 2] = t[None, None, :]
    return U, t, grid


# ---------------------------------------------------------------------------
# Conservative-form library and the true coefficients
# ---------------------------------------------------------------------------

#: The monomials of (w, u, v) of degree 1 and 2, as the library's input fields:
#: the conservative fluxes u w and v w are among them.
_VORT_MONOMIALS: tuple[tuple[int, ...], ...] = tuple(
    combo for degree in (1, 2) for combo in itertools.combinations_with_replacement((2, 0, 1), degree)
)
_VORT_STATE_NAMES = {0: "u", 1: "v", 2: "w"}
_VORT_MONOMIAL_NAMES: tuple[str, ...] = tuple(
    "*".join(_VORT_STATE_NAMES[i] for i in combo) for combo in _VORT_MONOMIALS
)
#: The field whose time derivative is identified: the vorticity.
_VORT_TARGETS: tuple[int, ...] = (_VORT_MONOMIAL_NAMES.index("w"),)


def vorticity_monomial_fields(U: np.ndarray) -> np.ndarray:
    """The monomials of ``(u, v, w)`` as fields, shape ``(..., 9)``."""

    U = np.asarray(U, dtype=float)
    return np.stack([np.prod(U[..., list(combo)], axis=-1) for combo in _VORT_MONOMIALS], axis=-1)


class VorticityWeakPDELibrary(DedupedWeakPDELibrary):
    """A weak library on the monomials of ``(u, v, w)``, identifying ``w`` alone.

    It takes the flow as ``(u, v, w)`` and expands it to the monomial fields
    itself; ``convert_u_dot_integral`` keeps only the vorticity.
    """

    def fit(self, x_full, y=None):
        return super().fit([vorticity_monomial_fields(x) for x in x_full], y)

    def transform(self, x_full):
        out = super().transform([vorticity_monomial_fields(x) for x in x_full])
        self.x_k = self.dx_k_j = self.dfx_k_j = None    # see ConservativeWeakPDELibrary
        return out

    def convert_u_dot_integral(self, u):
        full = np.asarray(super().convert_u_dot_integral(vorticity_monomial_fields(u)))
        return AxesArray(full[:, list(_VORT_TARGETS)], {"ax_sample": 0, "ax_coord": 1})


class WeightedVorticityWeakPDELibrary(WeightedWeakPDELibrary, VorticityWeakPDELibrary):
    """The weighted library on the vorticity target; its variance is the
    vorticity's, one component."""


def _build_function_library():
    """The function library on the monomial fields: the fields themselves."""

    return ps.PolynomialLibrary(degree=1, include_bias=False)


#: The vorticity equation in conservative form, against the library's feature
#: names (a monomial with pysindy's derivative suffix, axis 1 being x and 2 y).
_VORT_TRUE_TERMS: Dict[str, str] = {"w*u_1": "-1", "w*v_2": "-1", "w_11": "nu", "w_22": "nu"}


def build_true_vorticity_coefficients(feature_names: Sequence[str], *, nu: float) -> np.ndarray:
    """Analytic coefficients of the vorticity equation, shape (1, n_features).

    Raises if a term is missing from ``feature_names``, rather than score every
    rung against a truncated truth.
    """

    index = {name: j for j, name in enumerate(feature_names)}
    values = {"-1": -1.0, "nu": float(nu)}
    missing = [name for name in _VORT_TRUE_TERMS if name not in index]
    if missing:
        raise KeyError(f"The weak library does not provide {missing}.")
    coefficients = np.zeros((1, len(feature_names)))
    for name, symbol in _VORT_TRUE_TERMS.items():
        coefficients[0, index[name]] = values[symbol]
    return coefficients


def get_vorticity_feature_names(cfg, *, grid: np.ndarray, reference_trajectory: np.ndarray) -> tuple[str, ...]:
    """The weak library's feature names; they depend on neither support nor K."""

    library = VorticityWeakPDELibrary(
        function_library=_build_function_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=np.asarray(grid, dtype=float),
        is_uniform=True,
        include_bias=cfg.include_bias,
        include_interaction=False,
        K=2,
        p=cfg.p,
    )
    return get_library_feature_names(library, reference_trajectory, input_features=_VORT_MONOMIAL_NAMES)


def _make_weak_library(cfg, grid: np.ndarray, *, variance_field, weak_seed: int, whitener_mode: str = "full"):
    np.random.seed(weak_seed)
    H = list(cfg.H_xt) if cfg.H_xt is not None else [cfg.window / 4, cfg.window / 4, cfg.T / 20]
    common = dict(
        function_library=_build_function_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        is_uniform=True,
        include_bias=cfg.include_bias,
        include_interaction=False,
        H_xt=H,
        K=int(cfg.K) if cfg.K is not None else pde_coverage_test_function_count(grid, H, cfg.coverage),
        p=cfg.p,
    )
    if variance_field is None:
        return VorticityWeakPDELibrary(deduplicate=cfg.deduplicate, **common)
    return WeightedVorticityWeakPDELibrary(spatiotemporal_weights=variance_field, whitener_mode=whitener_mode,
                                           deduplicate=cfg.deduplicate, **common)


def _window_grid(cfg, grid: np.ndarray) -> np.ndarray:
    width = int(round(cfg.window / (cfg.L / cfg.N)))
    return np.ascontiguousarray(np.asarray(grid, dtype=float)[:width, :width])


# ---------------------------------------------------------------------------
# PART 1: multi-fidelity identification
# ---------------------------------------------------------------------------

@dataclass
class VorticityMultiTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Part I: one flow per run, one HF and ten LF windows, as the isothermal flow."""

    n_lf: int = 10
    n_hf: int = 1
    noise_lf_rel: float = 0.25
    noise_hf_rel: float = 0.01

    N: int = 128
    Nt: int = 1001
    L: float = 5.0
    T: float = 1.0
    window: float = 2.5     # edge of a window, a quarter of the domain (64 x 64 points)

    nu: float = 0.05

    derivative_order: int = 2
    include_bias: bool = False
    p: int = 2
    K: int | None = None
    deduplicate: bool = True
    H_xt: list[float] | None = None

    n_ensemble_models: int = 100
    n_jobs: int = 3
    seed_base: int = 0
    results_filename: str = "vorticity_part1_errors.csv"


def _dataset_batch(run_idx: int, cfg: VorticityMultiTrajectoryGLSConfig, noise_hf_abs: float,
                   noise_lf_abs: float, *, grid: np.ndarray, t: np.ndarray) -> MultiTrajectoryGLSData:
    rng = np.random.default_rng(cfg.seed_base + 100 * run_idx)
    U, _, _ = generate_vorticity_dataset(N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, nu=cfg.nu,
                                         seed=cfg.seed_base + run_idx * 1000)
    windows, width = ns_observation_windows(cfg.N, cfg.L, cfg.n_hf + cfg.n_lf, cfg.window, rng)
    clean = [U[idx] for idx in windows]
    return MultiTrajectoryGLSData(
        hf=[w + noise_hf_abs * rng.standard_normal(w.shape) for w in clean[:cfg.n_hf]],
        lf=[w + noise_lf_abs * rng.standard_normal(w.shape) for w in clean[cfg.n_hf:]],
        t_argument=t,
        metadata={"grid": np.ascontiguousarray(grid[:width, :width]),
                  "weak_seed": cfg.seed_base + 100_000 + run_idx, "windows": windows},
    )


def _fit_multi_trajectory_models(batch: MultiTrajectoryGLSData, cfg, optimizer_factory, *, t_argument,
                                 noise_hf_abs: float, noise_lf_abs: float,
                                 methods: Sequence[str] | None = None) -> Dict[str, np.ndarray]:
    """As for the isothermal flow: one library per group and weighting, the
    variance one scalar per fidelity group."""

    del t_argument
    grid = batch.metadata["grid"]
    weak_seed = int(batch.metadata["weak_seed"])
    shape = tuple(grid.shape[:-1])
    variances = {"hf": np.full(shape, noise_hf_abs**2), "lf": np.full(shape, noise_lf_abs**2)}
    libraries: Dict[Any, Any] = {}

    def library_for(variance_field, whitener_mode, sample):
        signature = _variance_signature(variance_field)
        key = None if signature is None else (signature, whitener_mode)
        if key is not None and key in libraries:
            return libraries[key]
        library = _make_weak_library(cfg, grid, variance_field=variance_field, weak_seed=weak_seed,
                                     whitener_mode=whitener_mode)
        library.fit([sample])
        if key is not None:
            libraries[key] = library
        return library

    def group_builder(fidelity: str, weighting: str):
        trajectories = batch.hf if fidelity == "hf" else batch.lf
        if weighting == "plain":
            variance_field = None
        elif weighting == "pooled":
            variance_field = np.ones(shape)    # fidelity-blind: only the correlations remain
        else:
            variance_field = variances[fidelity]
        whitener_mode = "diag" if weighting == "diag" else "full"
        theta, rhs = [], []
        for traj in trajectories:
            library = library_for(variance_field, whitener_mode, traj)
            theta.append(np.asarray(library.transform([traj])[0]))
            rhs.append(np.asarray(library.convert_u_dot_integral(traj)))
        return theta, rhs

    def fit_stacked(theta_blocks, rhs_blocks):
        return _fit_stacked_weak_system(theta_blocks, rhs_blocks, optimizer_factory)

    return assemble_weak_rungs(group_builder, fit_stacked, methods, seed=weak_seed)


def run_vorticity_multi_trajectory_gls_experiment(cfg: VorticityMultiTrajectoryGLSConfig):
    """Part I Monte Carlo; returns what run_multi_trajectory_gls_experiment does."""

    U_ref, t_ref, grid_ref = generate_vorticity_dataset(N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, nu=cfg.nu,
                                                        seed=cfg.seed_base)
    # Noise relative to the vorticity's spread on the reference flow, as the
    # isothermal flow uses the density's.
    state_std = float(np.std(U_ref[..., 2]))
    C_true = build_true_vorticity_coefficients(
        get_vorticity_feature_names(cfg, grid=grid_ref, reference_trajectory=U_ref), nu=cfg.nu)
    return run_multi_trajectory_gls_experiment(
        cfg,
        reference_state_std=lambda _cfg: state_std,
        dataset_builder=lambda run_idx, cfg, hf, lf: _dataset_batch(run_idx, cfg, hf, lf, grid=grid_ref, t=t_ref),
        true_coefficients=lambda _batch, _cfg: C_true,
        optimizer_factory=cfg.make_optimizer,
        fit_models_fn=_fit_multi_trajectory_models,
        progress_desc="MC vorticity MF",
    )


def vorticity_kappa_by_support(cfg, candidates, *, coefficients: np.ndarray, n_reference: int = 3,
                               weak_seed: int | None = None) -> pd.DataFrame:
    """Kappa per candidate support on windows of fixed reference flows, as
    ns_isothermal_kappa_by_support (sensitivities taken with respect to the
    monomial fields the library sees)."""

    seed = cfg.seed_base if weak_seed is None else int(weak_seed)
    references = []
    for j in range(int(n_reference)):
        U_ref, _, grid_ref = generate_vorticity_dataset(N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, nu=cfg.nu,
                                                        seed=cfg.seed_base + j)
        width = int(round(cfg.window / (cfg.L / cfg.N)))
        references.append((np.ascontiguousarray(U_ref[:width, :width]), _window_grid(cfg, grid_ref)))
    rows = []
    for H in candidates:
        medians, maxima = [], []
        for U_ref, grid_ref in references:
            library = _make_weak_library(replace(cfg, H_xt=list(H), K=None), grid_ref, variance_field=None,
                                         weak_seed=seed)
            library.fit_transform([U_ref])
            ratios = pde_weak_validity_ratio(
                library, pde_sensitivity_fields(library, vorticity_monomial_fields(U_ref), coefficients))
            medians.append(float(np.median(ratios)))
            maxima.append(float(np.max(ratios)))
        rows.append({"H_xt": list(H), "K": int(library.K), "kappa_median": float(np.median(medians)),
                     "kappa_lo": float(np.min(medians)), "kappa_hi": float(np.max(medians)),
                     "kappa_max": float(np.max(maxima))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART 2: heteroscedastic GLS
# ---------------------------------------------------------------------------

@dataclass
class VorticityIntraTrajectoryGLSConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Part II: ten windows of one flow, per-component state-dependent noise."""

    N: int = 128
    Nt: int = 1001
    L: float = 5.0
    T: float = 1.0
    n_traj: int = 10
    window: float = 2.5

    nu: float = 0.05

    # sigma_i(x, y, t) = noise_floor * s_i + noise_level * |u_i(x, y, t)|, as every Part II case
    noise_level: float = 0.15
    noise_floor: float = 1e-4

    derivative_order: int = 2
    include_bias: bool = False
    p: int = 2
    K: int | None = None
    deduplicate: bool = True
    H_xt: list[float] | None = None

    n_ensemble_models: int = 100
    n_jobs: int = 2
    results_filename: str = "vorticity_part2_errors.csv"


def vorticity_noise_variance(U_clean: np.ndarray, cfg) -> np.ndarray:
    """Pointwise noise variance per component (u, v, w), the two-component model
    of every Part II case, s_i being the spread of component i over the flow."""

    U_clean = np.asarray(U_clean, dtype=float)
    sigma = cfg.noise_floor * np.std(U_clean, axis=(0, 1, 2)) + cfg.noise_level * np.abs(U_clean)
    return np.maximum(sigma**2, 1e-12)


def vorticity_run_trajectories(cfg: VorticityIntraTrajectoryGLSConfig, rng: np.random.Generator, *,
                               observed: dict | None = None):
    """The cfg.n_traj windows of one Part II run, as ``(t, grid, clean, noisy, variance)``;
    the noise is drawn from the variance of the whole flow, one sensor measuring
    every window. ``observed`` receives the whole flow and the windows."""

    U, t, grid = generate_vorticity_dataset(N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, nu=cfg.nu,
                                            seed=int(rng.integers(2**31)))
    variance = vorticity_noise_variance(U, cfg)
    windows, width = ns_observation_windows(cfg.N, cfg.L, cfg.n_traj, cfg.window, rng)
    if observed is not None:
        observed.update(flow=U, windows=windows)
    clean, noisy, variances = [], [], []
    for idx in windows:
        clean.append(U[idx])
        variances.append(variance[idx])
        noisy.append(clean[-1] + np.sqrt(variances[-1]) * rng.standard_normal(size=clean[-1].shape))
    return t, np.ascontiguousarray(np.asarray(grid, dtype=float)[:width, :width]), clean, noisy, variances


def build_vorticity_artifacts(data, grid, t, target_variance, cfg, *, weak_seed: int,
                              true_coefficients: np.ndarray, H_xt=None, template=None) -> IntraTrajectoryGLSData:
    """The four Part II weightings for one window, on the vorticity library, every
    rung on the same test functions. ``target_variance`` is the vorticity's,
    shape (..., 1)."""

    H = list(H_xt) if H_xt is not None else (list(cfg.H_xt) if cfg.H_xt is not None
                                            else [cfg.window / 4, cfg.window / 4, cfg.T / 20])
    common = dict(
        function_library=_build_function_library(),
        derivative_order=cfg.derivative_order,
        spatiotemporal_grid=grid,
        is_uniform=True,
        include_bias=cfg.include_bias,
        include_interaction=False,
        H_xt=H,
        K=int(cfg.K) if cfg.K is not None else pde_coverage_test_function_count(grid, H, cfg.coverage),
        p=cfg.p,
        deduplicate=cfg.deduplicate,
    )

    def seeded(cls, **kwargs):
        np.random.seed(int(weak_seed))
        return cls(**common, **kwargs)

    libraries = {
        "No weighting": seeded(VorticityWeakPDELibrary),
        "Variance GLS": seeded(WeightedVorticityWeakPDELibrary, spatiotemporal_weights=target_variance),
        "Ones GLS": seeded(WeightedVorticityWeakPDELibrary, spatiotemporal_weights=np.ones(grid.shape[:-1])),
        "Variance diag": seeded(WeightedVorticityWeakPDELibrary, spatiotemporal_weights=target_variance,
                                whitener_mode="diag"),
    }
    _share_test_functions(libraries.values(), template or libraries["No weighting"])
    return IntraTrajectoryGLSData(data=data, t_argument=t, libraries=libraries,
                                  true_coefficients=np.asarray(true_coefficients))


def _make_gls_artifacts(run_idx: int, cfg, rng: np.random.Generator, *, true_coefficients: np.ndarray):
    t, grid, clean, noisy, variances = vorticity_run_trajectories(cfg, rng)
    out, template = [], None
    for data, variance in zip(noisy, variances):
        out.append(build_vorticity_artifacts(
            data, grid, t, variance[..., 2:3], cfg, weak_seed=cfg.seed_base + 1000 + run_idx,
            true_coefficients=true_coefficients, template=template))
        template = out[0].libraries["No weighting"]
    return out


def run_vorticity_intra_trajectory_gls_experiment(cfg: VorticityIntraTrajectoryGLSConfig):
    """Part II Monte Carlo; returns what run_intra_trajectory_gls_experiment does."""

    U_ref, _, grid_ref = generate_vorticity_dataset(N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, nu=cfg.nu,
                                                    seed=cfg.seed_base)
    C_true = build_true_vorticity_coefficients(
        get_vorticity_feature_names(cfg, grid=grid_ref, reference_trajectory=U_ref), nu=cfg.nu)

    def builder(run_idx: int, cfg):
        return _make_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx),
                                   true_coefficients=C_true)

    return run_intra_trajectory_gls_experiment(cfg, run_builder=builder, progress_desc="MC vorticity GLS")
