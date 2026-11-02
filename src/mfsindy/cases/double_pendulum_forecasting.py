"""Double-pendulum trajectory generation and weighted weak-SINDy utilities."""

from __future__ import annotations

import os
import pickle
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps

from mfsindy.experiments import (
    PART1_METHODS,
    BICEnsembleSTLSQ,
    EnsembleConfigMixin,
    MonteCarloConfig,
    MultiTrajectoryGLSData,
)
from mfsindy.experiments.selection import select_rungs, support_key, support_variants
from mfsindy.weighted_weak_pde_library import WeightedWeakPDELibrary

DEFAULT_THETA_RANGE = (-np.pi/3, np.pi/3)
DEFAULT_OMEGA_RANGE = (-0.0, 0.0)
DEFAULT_HF_NOISE_SCALE = 0.01
DEFAULT_LF_NOISE_SCALE = 0.10
DEFAULT_POLY_DEGREE = 3
DEFAULT_LIBRARY_KIND = "true"


@dataclass
class DoublePendulumObservationSet:
    """Clean/noisy trajectory bundle for one fidelity group."""

    clean: np.ndarray
    noisy: np.ndarray
    variance: np.ndarray


@dataclass
class DoublePendulumForecastingDataset:
    """HF/LF dataset used in the forecasting notebook."""

    t: np.ndarray
    hf: DoublePendulumObservationSet
    lf: DoublePendulumObservationSet
    hf_noise_scale: float
    lf_noise_scale: float
    m1: float
    m2: float
    l1: float
    l2: float
    g: float


@dataclass
class WeightedWeakEnsembleFit:
    """Stacked weighted weak-SINDy fit."""

    coefficients: np.ndarray
    ensemble_coefficients: np.ndarray
    feature_names: list[str]


@dataclass
class EnsembleIntervalSummary:
    """Median and quantile envelope for one ensemble output."""

    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class DoublePendulumModelMetrics:
    """Forecast/derivative metrics for one fitted fidelity model."""

    forecast_rmse: float
    forecast_width: float
    forecast_coverage: float
    forecast_valid_fraction: float
    derivative_rmse: float
    derivative_width: float
    derivative_coverage: float
    derivative_valid_fraction: float


@dataclass
class DoublePendulumEvaluation:
    """Future-trajectory evaluation for a set of fitted fidelity models."""

    t_future: np.ndarray
    true_future: np.ndarray
    true_future_derivatives: np.ndarray
    forecast_summaries: dict[str, EnsembleIntervalSummary]
    derivative_summaries: dict[str, EnsembleIntervalSummary]
    metrics: dict[str, DoublePendulumModelMetrics]


@contextmanager
def _temporary_numpy_seed(seed: int | None):
    """Temporarily set the global NumPy RNG seed and restore it afterwards."""

    if seed is None:
        yield
        return

    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(int(seed))  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def sample_double_pendulum_initial_condition(
    rng: np.random.Generator,
    *,
    theta_range: tuple[float, float] = DEFAULT_THETA_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
) -> np.ndarray:
    """Sample one initial condition for the double pendulum.

    The default ranges keep the angles in a moderately nonlinear regime and the
    angular velocities bounded:

    - theta_1, theta_2 in [-pi/3, pi/3]
    - omega_1, omega_2 in [-1, 1]
    """

    theta_min, theta_max = theta_range
    omega_min, omega_max = omega_range
    return np.array(
        [
            rng.uniform(theta_min, theta_max),
            rng.uniform(omega_min, omega_max),
            rng.uniform(theta_min, theta_max),
            rng.uniform(omega_min, omega_max),
        ],
        dtype=float,
    )


def _double_pendulum_rhs(
    state: np.ndarray,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> np.ndarray:
    """Return the time derivative of the planar double pendulum."""

    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1

    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    den2 = (l2 / l1) * den1

    dtheta1 = omega1
    dtheta2 = omega2
    domega1 = (
        m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * l2 * omega2**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / den1
    domega2 = (
        -m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2)
        * (
            g * np.sin(theta1) * np.cos(delta)
            - l1 * omega1**2 * np.sin(delta)
            - g * np.sin(theta2)
        )
    ) / den2

    return np.array([dtheta1, domega1, dtheta2, domega2], dtype=float)


def _double_pendulum_denominator(
    theta1: np.ndarray,
    theta2: np.ndarray,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
) -> np.ndarray:
    """Return the shared denominator appearing in the double-pendulum RHS."""

    delta = theta2 - theta1
    return (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2


def _rk4_step(state: np.ndarray, dt: float, **rhs_kwargs) -> np.ndarray:
    """Advance one Runge-Kutta step."""

    k1 = _double_pendulum_rhs(state, **rhs_kwargs)
    k2 = _double_pendulum_rhs(state + 0.5 * dt * k1, **rhs_kwargs)
    k3 = _double_pendulum_rhs(state + 0.5 * dt * k2, **rhs_kwargs)
    k4 = _double_pendulum_rhs(state + dt * k3, **rhs_kwargs)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_double_pendulum_trajectory(
    *,
    y0: np.ndarray | None = None,
    t_final: float = 12.0,
    dt: float = 0.01,
    theta_range: tuple[float, float] = DEFAULT_THETA_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
    seed: int | None = None,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one clean double-pendulum trajectory.

    Parameters
    ----------
    y0
        Initial condition `[theta_1, omega_1, theta_2, omega_2]`. If omitted,
        it is sampled from the default angle and velocity ranges.
    t_final
        Final integration time.
    dt
        Time step.
    theta_range, omega_range
        Sampling ranges used only when `y0` is not provided.
    seed
        RNG seed used for initial-condition sampling.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if t_final <= 0.0:
        raise ValueError("t_final must be positive.")

    rng = np.random.default_rng(seed)
    if y0 is None:
        y0 = sample_double_pendulum_initial_condition(
            rng,
            theta_range=theta_range,
            omega_range=omega_range,
        )
    else:
        y0 = np.asarray(y0, dtype=float)

    n_steps = int(np.round(t_final / dt)) + 1
    t = np.linspace(0.0, t_final, n_steps)
    states = np.zeros((n_steps, 4), dtype=float)
    states[0] = y0

    rhs_kwargs = {"m1": m1, "m2": m2, "l1": l1, "l2": l2, "g": g}
    for k in range(1, n_steps):
        states[k] = _rk4_step(states[k - 1], dt, **rhs_kwargs)

    return t, states


def compute_double_pendulum_derivatives(
    states: np.ndarray,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> np.ndarray:
    """Evaluate the true double-pendulum derivatives along a trajectory."""

    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("states must have shape (n_steps, 4).")

    rhs_kwargs = {"m1": m1, "m2": m2, "l1": l1, "l2": l2, "g": g}
    return np.stack(
        [_double_pendulum_rhs(state, **rhs_kwargs) for state in states],
        axis=0,
    )


def generate_double_pendulum_trajectories(
    n_traj: int = 1,
    *,
    t_final: float = 12.0,
    dt: float = 0.01,
    theta_range: tuple[float, float] = DEFAULT_THETA_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
    seed: int = 0,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate multiple clean double-pendulum trajectories.

    Returns
    -------
    t
        Shared time grid with shape `(n_steps,)`.
    trajectories
        Array with shape `(n_traj, n_steps, 4)`.
    """

    if n_traj < 1:
        raise ValueError("n_traj must be at least 1.")

    rng = np.random.default_rng(seed)
    trajectories: list[np.ndarray] = []
    t_ref: np.ndarray | None = None

    for _ in range(n_traj):
        y0 = sample_double_pendulum_initial_condition(
            rng,
            theta_range=theta_range,
            omega_range=omega_range,
        )
        t, states = simulate_double_pendulum_trajectory(
            y0=y0,
            t_final=t_final,
            dt=dt,
            theta_range=theta_range,
            omega_range=omega_range,
            seed=None,
            m1=m1,
            m2=m2,
            l1=l1,
            l2=l2,
            g=g,
        )
        trajectories.append(states)
        if t_ref is None:
            t_ref = t

    return t_ref, np.stack(trajectories, axis=0)


def omega_variance_profile(
    states: np.ndarray,
    *,
    noise_scale: float,
    min_variance: float = 1e-10,
) -> np.ndarray:
    """Return the time-dependent variance profile sigma(t)^2.

    The profile is scalar in time and shared across state components:

        sigma(t) = noise_scale * ||omega(t)||,

    where ||omega(t)|| = sqrt(omega_1(t)^2 + omega_2(t)^2).
    """

    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("states must have shape (n_steps, 4).")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative.")

    omega_mag = np.linalg.norm(states[:, [1, 3]], axis=1)
    sigma_t = noise_scale * np.maximum(omega_mag, 1e-12)
    variance_t = np.maximum(sigma_t**2, min_variance)
    return variance_t[:, None]


def add_omega_magnitude_noise(
    states: np.ndarray,
    *,
    noise_scale: float,
    seed: int | None = None,
    min_variance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise with sigma(t) = noise_scale * ||omega(t)||."""

    variance = omega_variance_profile(
        states,
        noise_scale=noise_scale,
        min_variance=min_variance,
    )
    rng = np.random.default_rng(seed)
    noisy_states = states + rng.normal(
        loc=0.0,
        scale=np.sqrt(variance),
        size=states.shape,
    )
    return noisy_states, variance


def build_double_pendulum_dataset(
    *,
    n_hf: int = 1,
    n_lf: int = 5,
    t_final: float = 10.0,
    dt: float = 0.01,
    hf_noise_scale: float = DEFAULT_HF_NOISE_SCALE,
    lf_noise_scale: float = DEFAULT_LF_NOISE_SCALE,
    seed: int = 0,
    theta_range: tuple[float, float] = DEFAULT_THETA_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> DoublePendulumForecastingDataset:
    """Generate HF/LF observations for the forecasting benchmark.

    All sensors observe the same clean trajectory. This matches the
    forecasting setup in which one underlying path is measured once at high
    fidelity and several times at low fidelity.
    """

    t_hf, clean_traj = simulate_double_pendulum_trajectory(
        t_final=t_final,
        dt=dt,
        theta_range=theta_range,
        omega_range=omega_range,
        seed=seed,
        m1=m1,
        m2=m2,
        l1=l1,
        l2=l2,
        g=g,
    )
    hf_clean = np.repeat(clean_traj[None, :, :], n_hf, axis=0)
    lf_clean = np.repeat(clean_traj[None, :, :], n_lf, axis=0)

    hf_noisy: list[np.ndarray] = []
    hf_variance: list[np.ndarray] = []
    for idx, traj in enumerate(hf_clean):
        noisy_traj, variance = add_omega_magnitude_noise(
            traj,
            noise_scale=hf_noise_scale,
            seed=seed + 10_000 + idx,
        )
        hf_noisy.append(noisy_traj)
        hf_variance.append(variance)

    lf_noisy: list[np.ndarray] = []
    lf_variance: list[np.ndarray] = []
    for idx, traj in enumerate(lf_clean):
        noisy_traj, variance = add_omega_magnitude_noise(
            traj,
            noise_scale=lf_noise_scale,
            seed=seed + 20_000 + idx,
        )
        lf_noisy.append(noisy_traj)
        lf_variance.append(variance)

    return DoublePendulumForecastingDataset(
        t=t_hf,
        hf=DoublePendulumObservationSet(
            clean=hf_clean,
            noisy=np.stack(hf_noisy, axis=0),
            variance=np.stack(hf_variance, axis=0),
        ),
        lf=DoublePendulumObservationSet(
            clean=lf_clean,
            noisy=np.stack(lf_noisy, axis=0),
            variance=np.stack(lf_variance, axis=0),
        ),
        hf_noise_scale=hf_noise_scale,
        lf_noise_scale=lf_noise_scale,
        m1=m1,
        m2=m2,
        l1=l1,
        l2=l2,
        g=g,
    )


def build_double_pendulum_feature_library(
    degree: int = DEFAULT_POLY_DEGREE,
    *,
    library_kind: str = DEFAULT_LIBRARY_KIND,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
) -> ps.BaseFeatureLibrary:
    """Return either the true or polynomial library for the double pendulum.

    The `degree` argument is kept only for API compatibility with the earlier
    polynomial baseline.
    """

    if library_kind == "polynomial":
        return ps.PolynomialLibrary(
            degree=degree,
            include_bias=False,
            include_interaction=True,
            interaction_only=False,
        )
    if library_kind != "true":
        raise ValueError("library_kind must be either 'true' or 'polynomial'.")

    _ = degree

    def omega1_feature(theta1, omega1, theta2, omega2):
        return omega1

    def omega2_feature(theta1, omega1, theta2, omega2):
        return omega2

    def sin_theta1_over_den(theta1, omega1, theta2, omega2):
        den = _double_pendulum_denominator(theta1, theta2, m1=m1, m2=m2, l1=l1)
        return np.sin(theta1) / den

    def sin_theta2_over_den(theta1, omega1, theta2, omega2):
        den = _double_pendulum_denominator(theta1, theta2, m1=m1, m2=m2, l1=l1)
        return np.sin(theta2) / den

    def sin_theta1_cos_delta_over_den(theta1, omega1, theta2, omega2):
        delta = theta2 - theta1
        den = _double_pendulum_denominator(theta1, theta2, m1=m1, m2=m2, l1=l1)
        return np.sin(theta1) * np.cos(delta) / den

    def sin_theta2_cos_delta_over_den(theta1, omega1, theta2, omega2):
        delta = theta2 - theta1
        den = _double_pendulum_denominator(theta1, theta2, m1=m1, m2=m2, l1=l1)
        return np.sin(theta2) * np.cos(delta) / den

    def omega1_sq_sin_delta_cos_delta_over_den(theta1, omega1, theta2, omega2):
        delta = theta2 - theta1
        den = _double_pendulum_denominator(theta1, theta2, m1=m1, m2=m2, l1=l1)
        return omega1**2 * np.sin(delta) * np.cos(delta) / den

    def omega2_sq_sin_delta_over_den(theta1, omega1, theta2, omega2):
        delta = theta2 - theta1
        den = _double_pendulum_denominator(theta1, theta2, m1=m1, m2=m2, l1=l1)
        return omega2**2 * np.sin(delta) / den

    def omega1_sq_sin_delta_over_den(theta1, omega1, theta2, omega2):
        delta = theta2 - theta1
        den = _double_pendulum_denominator(theta1, theta2, m1=m1, m2=m2, l1=l1)
        return omega1**2 * np.sin(delta) / den

    def omega2_sq_sin_delta_cos_delta_over_den(theta1, omega1, theta2, omega2):
        delta = theta2 - theta1
        den = _double_pendulum_denominator(theta1, theta2, m1=m1, m2=m2, l1=l1)
        return omega2**2 * np.sin(delta) * np.cos(delta) / den

    library_functions = [
        omega1_feature,
        omega2_feature,
        sin_theta1_over_den,
        sin_theta2_over_den,
        sin_theta1_cos_delta_over_den,
        sin_theta2_cos_delta_over_den,
        omega1_sq_sin_delta_cos_delta_over_den,
        omega2_sq_sin_delta_over_den,
        omega1_sq_sin_delta_over_den,
        omega2_sq_sin_delta_cos_delta_over_den,
    ]
    function_names = [
        lambda theta1, omega1, theta2, omega2: omega1,
        lambda theta1, omega1, theta2, omega2: omega2,
        lambda theta1, omega1, theta2, omega2: f"sin({theta1})/den",
        lambda theta1, omega1, theta2, omega2: f"sin({theta2})/den",
        lambda theta1, omega1, theta2, omega2: f"sin({theta1}) cos({theta2} - {theta1})/den",
        lambda theta1, omega1, theta2, omega2: f"sin({theta2}) cos({theta2} - {theta1})/den",
        lambda theta1, omega1, theta2, omega2: f"{omega1}^2 sin({theta2} - {theta1}) cos({theta2} - {theta1})/den",
        lambda theta1, omega1, theta2, omega2: f"{omega2}^2 sin({theta2} - {theta1})/den",
        lambda theta1, omega1, theta2, omega2: f"{omega1}^2 sin({theta2} - {theta1})/den",
        lambda theta1, omega1, theta2, omega2: f"{omega2}^2 sin({theta2} - {theta1}) cos({theta2} - {theta1})/den",
    ]
    return ps.CustomLibrary(
        library_functions=library_functions,
        function_names=function_names,
        interaction_only=True,
        include_bias=False,
    )


def remove_fidelity_scaling_from_variance(
    variance_fields: np.ndarray,
    *,
    noise_scale: float,
    min_variance: float = 1e-10,
) -> np.ndarray:
    """Remove the global fidelity scale while preserving the time profile."""

    variance_fields = np.asarray(variance_fields, dtype=float)
    if noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive.")

    normalized = variance_fields / float(noise_scale**2)
    return np.maximum(normalized, min_variance)


def make_weighted_double_pendulum_weak_library(
    t_grid: np.ndarray,
    variance_field: np.ndarray,
    *,
    poly_degree: int = DEFAULT_POLY_DEGREE,
    library_kind: str = DEFAULT_LIBRARY_KIND,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    weak_seed: int = 0,
    n_weak_cells: int = 200,
    weak_window_length: float = 0.5,
    p: int = 4,
    include_bias: bool = False,
) -> WeightedWeakPDELibrary:
    """Create the weighted weak library for one noisy trajectory."""

    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.ndim == 1:
        t_grid = t_grid[:, None]

    return WeightedWeakPDELibrary(
        function_library=build_double_pendulum_feature_library(
            degree=poly_degree,
            library_kind=library_kind,
            m1=m1,
            m2=m2,
            l1=l1,
        ),
        spatiotemporal_grid=t_grid,
        spatiotemporal_weights=variance_field,
        include_bias=include_bias,
        include_interaction=False,
        K=n_weak_cells,
        H_xt=weak_window_length,
        p=p,
    )


def _median_ensemble_coefficients(
    optimizer: ps.BaseOptimizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the median ensemble coefficients and the full ensemble stack."""

    coef_list = getattr(optimizer, "coef_list", None)
    if coef_list:
        ensemble_coefficients = np.asarray(coef_list, dtype=float)
        if ensemble_coefficients.ndim == 3:
            coefficients = np.median(ensemble_coefficients, axis=0)
            return coefficients, ensemble_coefficients

    coefficients = np.asarray(optimizer.coef_, dtype=float)
    return coefficients, coefficients[None, ...]


def fit_double_pendulum_weighted_weak_model(
    trajectories: np.ndarray,
    variance_fields: np.ndarray,
    t_grid: np.ndarray,
    *,
    poly_degree: int = DEFAULT_POLY_DEGREE,
    library_kind: str = DEFAULT_LIBRARY_KIND,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    stlsq_threshold: float = 0.3,
    stlsq_alpha: float = 1e-6,
    n_ensemble_models: int = 25,
    weak_seed: int = 0,
    n_weak_cells: int = 200,
    weak_window_length: float = 0.5,
    p: int = 4,
) -> WeightedWeakEnsembleFit:
    """Fit one weighted weak ensemble model from a set of trajectories."""

    trajectories = np.asarray(trajectories, dtype=float)
    variance_fields = np.asarray(variance_fields, dtype=float)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 4:
        raise ValueError("trajectories must have shape (n_traj, n_steps, 4).")
    if variance_fields.shape[:2] != trajectories.shape[:2]:
        raise ValueError("variance_fields must match the first two trajectory dimensions.")

    theta_blocks: list[np.ndarray] = []
    rhs_blocks: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for traj, variance_field in zip(trajectories, variance_fields, strict=True):
        weak_lib = make_weighted_double_pendulum_weak_library(
            t_grid,
            variance_field,
            poly_degree=poly_degree,
            library_kind=library_kind,
            m1=m1,
            m2=m2,
            l1=l1,
            weak_seed=weak_seed,
            n_weak_cells=n_weak_cells,
            weak_window_length=weak_window_length,
            p=p,
        )
        theta_blocks.append(np.asarray(weak_lib.fit_transform([traj])[0], dtype=float))
        rhs_blocks.append(np.asarray(weak_lib.convert_u_dot_integral(traj), dtype=float))
        if feature_names is None:
            feature_names = weak_lib.get_feature_names(
                input_features=["theta1", "omega1", "theta2", "omega2"]
            )

    optimizer = ps.EnsembleOptimizer(
        ps.STLSQ(threshold=stlsq_threshold, alpha=stlsq_alpha),
        n_models=n_ensemble_models,
        bagging=True,
    )
    optimizer.fit(np.vstack(theta_blocks), np.vstack(rhs_blocks))
    coefficients, ensemble_coefficients = _median_ensemble_coefficients(optimizer)
    return WeightedWeakEnsembleFit(
        coefficients=coefficients,
        ensemble_coefficients=ensemble_coefficients,
        feature_names=feature_names or [],
    )


def fit_double_pendulum_hf_lf_models(
    dataset: DoublePendulumForecastingDataset,
    *,
    poly_degree: int = DEFAULT_POLY_DEGREE,
    library_kind: str = DEFAULT_LIBRARY_KIND,
    stlsq_threshold: float = 0.3,
    stlsq_alpha: float = 1e-6,
    n_ensemble_models: int = 25,
    weak_seed: int = 0,
    n_weak_cells: int = 200,
    weak_window_length: float = 0.5,
    p: int = 4,
) -> dict[str, WeightedWeakEnsembleFit]:
    """Fit HF, LF, and MF weighted weak-SINDy models."""

    common_kwargs = {
        "t_grid": dataset.t,
        "poly_degree": poly_degree,
        "library_kind": library_kind,
        "m1": dataset.m1,
        "m2": dataset.m2,
        "l1": dataset.l1,
        "stlsq_threshold": stlsq_threshold,
        "stlsq_alpha": stlsq_alpha,
        "n_ensemble_models": n_ensemble_models,
        "weak_seed": weak_seed,
        "n_weak_cells": n_weak_cells,
        "weak_window_length": weak_window_length,
        "p": p,
    }
    return {
        "HF": fit_double_pendulum_weighted_weak_model(
            dataset.hf.noisy,
            dataset.hf.variance,
            **common_kwargs,
        ),
        "LF": fit_double_pendulum_weighted_weak_model(
            dataset.lf.noisy,
            dataset.lf.variance,
            **common_kwargs,
        ),
        "MF": fit_double_pendulum_weighted_weak_model(
            np.concatenate([dataset.hf.noisy, dataset.lf.noisy], axis=0),
            np.concatenate([dataset.hf.variance, dataset.lf.variance], axis=0),
            **common_kwargs,
        ),
    }


def format_weighted_weak_equations(
    coefficients: np.ndarray,
    feature_names: list[str],
    *,
    state_names: tuple[str, str, str, str] = (
        "theta1_dot",
        "omega1_dot",
        "theta2_dot",
        "omega2_dot",
    ),
    precision: int = 3,
    zero_tol: float = 1e-12,
) -> list[str]:
    """Format a coefficient matrix as readable equations."""

    coefficients = np.asarray(coefficients, dtype=float)
    equations: list[str] = []

    for state_name, row in zip(state_names, coefficients, strict=True):
        terms: list[str] = []
        for coef, feature_name in zip(row, feature_names, strict=True):
            if abs(coef) <= zero_tol:
                continue
            terms.append(f"{coef:.{precision}f} {feature_name}")
        rhs = " + ".join(terms).replace("+ -", "- ")
        equations.append(f"{state_name} = {rhs or '0'}")
    return equations


def _pointwise_feature_vector(
    state: np.ndarray,
    feature_library: ps.BaseFeatureLibrary,
) -> np.ndarray:
    """Evaluate the pointwise SINDy library on one state vector."""

    state = np.asarray(state, dtype=float)[None, :]
    return np.asarray(feature_library.transform(state), dtype=float)[0]


def simulate_ensemble_sindy_forecast(
    ensemble_coefficients: np.ndarray,
    y0: np.ndarray,
    *,
    poly_degree: int = DEFAULT_POLY_DEGREE,
    library_kind: str = DEFAULT_LIBRARY_KIND,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    t_final: float = 10.0,
    dt: float = 0.01,
    max_abs_state: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out each ensemble member as a deterministic SINDy model."""

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if t_final <= 0.0:
        raise ValueError("t_final must be positive.")

    ensemble_coefficients = np.asarray(ensemble_coefficients, dtype=float)
    if ensemble_coefficients.ndim == 2:
        ensemble_coefficients = ensemble_coefficients[None, ...]
    if ensemble_coefficients.ndim != 3 or ensemble_coefficients.shape[1] != 4:
        raise ValueError("ensemble_coefficients must have shape (n_models, 4, n_features).")

    y0 = np.asarray(y0, dtype=float)
    n_steps = int(np.round(t_final / dt)) + 1
    t = np.linspace(0.0, t_final, n_steps)
    forecasts = np.zeros((ensemble_coefficients.shape[0], n_steps, 4), dtype=float)
    forecasts[:, 0, :] = y0

    feature_library = build_double_pendulum_feature_library(
        degree=poly_degree,
        library_kind=library_kind,
        m1=m1,
        m2=m2,
        l1=l1,
    )
    feature_library.fit(np.zeros((2, 4), dtype=float))

    def rhs(state: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        theta = _pointwise_feature_vector(state, feature_library)
        return theta @ coefficients.T

    for model_idx, coefficients in enumerate(ensemble_coefficients):
        for step_idx in range(1, n_steps):
            state = forecasts[model_idx, step_idx - 1]
            if not np.all(np.isfinite(state)):
                forecasts[model_idx, step_idx:] = np.nan
                break
            with np.errstate(over="ignore", invalid="ignore"):
                k1 = rhs(state, coefficients)
                k2 = rhs(state + 0.5 * dt * k1, coefficients)
                k3 = rhs(state + 0.5 * dt * k2, coefficients)
                k4 = rhs(state + dt * k3, coefficients)
                next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.all(np.isfinite(next_state)) or np.max(np.abs(next_state)) > max_abs_state:
                forecasts[model_idx, step_idx:] = np.nan
                break
            forecasts[model_idx, step_idx] = next_state

    return t, forecasts


def evaluate_ensemble_sindy_derivatives(
    states: np.ndarray,
    ensemble_coefficients: np.ndarray,
    *,
    poly_degree: int = DEFAULT_POLY_DEGREE,
    library_kind: str = DEFAULT_LIBRARY_KIND,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
) -> np.ndarray:
    """Evaluate all ensemble members on a given trajectory."""

    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("states must have shape (n_steps, 4).")

    ensemble_coefficients = np.asarray(ensemble_coefficients, dtype=float)
    if ensemble_coefficients.ndim == 2:
        ensemble_coefficients = ensemble_coefficients[None, ...]
    if ensemble_coefficients.ndim != 3 or ensemble_coefficients.shape[1] != 4:
        raise ValueError("ensemble_coefficients must have shape (n_models, 4, n_features).")

    feature_library = build_double_pendulum_feature_library(
        degree=poly_degree,
        library_kind=library_kind,
        m1=m1,
        m2=m2,
        l1=l1,
    )
    feature_library.fit(np.zeros((2, 4), dtype=float))
    theta = np.asarray(feature_library.transform(states), dtype=float)
    return np.einsum("sf,mnf->msn", theta, ensemble_coefficients)


def summarize_ensemble_forecast(
    ensemble_forecast: np.ndarray,
    *,
    quantile_band: tuple[float, float] = (0.1, 0.9),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the median and uncertainty envelope of an ensemble forecast."""

    ensemble_forecast = np.asarray(ensemble_forecast, dtype=float)
    if ensemble_forecast.ndim != 3:
        raise ValueError("ensemble_forecast must have shape (n_models, n_steps, n_states).")

    q_low, q_high = quantile_band
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        median = np.nanmedian(ensemble_forecast, axis=0)
        lower = np.nanquantile(ensemble_forecast, q_low, axis=0)
        upper = np.nanquantile(ensemble_forecast, q_high, axis=0)
    return median, lower, upper


def _compute_interval_metrics(
    truth: np.ndarray,
    summary: EnsembleIntervalSummary,
) -> tuple[float, float, float, float]:
    """Compute RMSE, band width, coverage, and valid fraction."""

    truth = np.asarray(truth, dtype=float)
    valid = (
        np.isfinite(truth)
        & np.isfinite(summary.median)
        & np.isfinite(summary.lower)
        & np.isfinite(summary.upper)
    )
    valid_fraction = float(np.mean(valid))

    if not np.any(valid):
        return np.nan, np.nan, np.nan, valid_fraction

    rmse = float(np.sqrt(np.mean((summary.median[valid] - truth[valid]) ** 2)))
    width = float(np.nanmean(summary.upper - summary.lower))
    coverage = float(
        np.mean(
            (truth[valid] >= summary.lower[valid])
            & (truth[valid] <= summary.upper[valid])
        )
    )
    return rmse, width, coverage, valid_fraction


def evaluate_double_pendulum_models(
    dataset: DoublePendulumForecastingDataset,
    fit_results: dict[str, WeightedWeakEnsembleFit],
    *,
    forecast_horizon: float = 5.0,
    forecast_dt: float | None = None,
    quantile_band: tuple[float, float] = (0.1, 0.9),
    poly_degree: int = DEFAULT_POLY_DEGREE,
    library_kind: str = DEFAULT_LIBRARY_KIND,
    max_abs_state: float = 50.0,
) -> DoublePendulumEvaluation:
    """Evaluate fitted HF/LF/MF models on future states and derivatives."""

    if forecast_dt is None:
        if dataset.t.size < 2:
            raise ValueError("dataset.t must contain at least two time samples.")
        forecast_dt = float(dataset.t[1] - dataset.t[0])

    y0_forecast = np.asarray(dataset.hf.clean[0, -1], dtype=float)
    t_future, true_future = simulate_double_pendulum_trajectory(
        y0=y0_forecast,
        t_final=forecast_horizon,
        dt=forecast_dt,
        m1=dataset.m1,
        m2=dataset.m2,
        l1=dataset.l1,
        l2=dataset.l2,
        g=dataset.g,
    )
    true_future_derivatives = compute_double_pendulum_derivatives(
        true_future,
        m1=dataset.m1,
        m2=dataset.m2,
        l1=dataset.l1,
        l2=dataset.l2,
        g=dataset.g,
    )

    forecast_summaries: dict[str, EnsembleIntervalSummary] = {}
    derivative_summaries: dict[str, EnsembleIntervalSummary] = {}
    metrics: dict[str, DoublePendulumModelMetrics] = {}

    for model_name, result in fit_results.items():
        _, ensemble_forecast = simulate_ensemble_sindy_forecast(
            result.ensemble_coefficients,
            y0_forecast,
            poly_degree=poly_degree,
            library_kind=library_kind,
            m1=dataset.m1,
            m2=dataset.m2,
            l1=dataset.l1,
            t_final=forecast_horizon,
            dt=forecast_dt,
            max_abs_state=max_abs_state,
        )
        forecast_summary = EnsembleIntervalSummary(
            *summarize_ensemble_forecast(
                ensemble_forecast,
                quantile_band=quantile_band,
            )
        )
        forecast_summaries[model_name] = forecast_summary

        ensemble_derivatives = evaluate_ensemble_sindy_derivatives(
            true_future,
            result.ensemble_coefficients,
            poly_degree=poly_degree,
            library_kind=library_kind,
            m1=dataset.m1,
            m2=dataset.m2,
            l1=dataset.l1,
        )
        derivative_summary = EnsembleIntervalSummary(
            *summarize_ensemble_forecast(
                ensemble_derivatives,
                quantile_band=quantile_band,
            )
        )
        derivative_summaries[model_name] = derivative_summary

        (
            forecast_rmse,
            forecast_width,
            forecast_coverage,
            forecast_valid_fraction,
        ) = _compute_interval_metrics(true_future, forecast_summary)
        (
            derivative_rmse,
            derivative_width,
            derivative_coverage,
            derivative_valid_fraction,
        ) = _compute_interval_metrics(true_future_derivatives, derivative_summary)

        metrics[model_name] = DoublePendulumModelMetrics(
            forecast_rmse=forecast_rmse,
            forecast_width=forecast_width,
            forecast_coverage=forecast_coverage,
            forecast_valid_fraction=forecast_valid_fraction,
            derivative_rmse=derivative_rmse,
            derivative_width=derivative_width,
            derivative_coverage=derivative_coverage,
            derivative_valid_fraction=derivative_valid_fraction,
        )

    return DoublePendulumEvaluation(
        t_future=t_future,
        true_future=true_future,
        true_future_derivatives=true_future_derivatives,
        forecast_summaries=forecast_summaries,
        derivative_summaries=derivative_summaries,
        metrics=metrics,
    )


def benchmark_double_pendulum_fidelities(
    seeds: list[int] | np.ndarray,
    *,
    n_hf: int = 1,
    n_lf: int = 5,
    t_final: float = 10.0,
    dt: float = 0.01,
    hf_noise_scale: float = DEFAULT_HF_NOISE_SCALE,
    lf_noise_scale: float = DEFAULT_LF_NOISE_SCALE,
    theta_range: tuple[float, float] = DEFAULT_THETA_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
    poly_degree: int = DEFAULT_POLY_DEGREE,
    library_kind: str = DEFAULT_LIBRARY_KIND,
    stlsq_threshold: float = 0.3,
    stlsq_alpha: float = 1e-6,
    n_ensemble_models: int = 25,
    n_weak_cells: int = 200,
    weak_window_length: float = 0.5,
    p: int = 4,
    forecast_horizon: float = 5.0,
    forecast_dt: float | None = None,
    quantile_band: tuple[float, float] = (0.1, 0.9),
    max_abs_state: float = 50.0,
) -> list[dict[str, float | int | str]]:
    """Run the current HF/LF/MF benchmark for a collection of seeds."""

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        seed_int = int(seed)
        dataset = build_double_pendulum_dataset(
            n_hf=n_hf,
            n_lf=n_lf,
            t_final=t_final,
            dt=dt,
            hf_noise_scale=hf_noise_scale,
            lf_noise_scale=lf_noise_scale,
            seed=seed_int,
            theta_range=theta_range,
            omega_range=omega_range,
            m1=m1,
            m2=m2,
            l1=l1,
            l2=l2,
            g=g,
        )
        fit_results = fit_double_pendulum_hf_lf_models(
            dataset,
            poly_degree=poly_degree,
            library_kind=library_kind,
            stlsq_threshold=stlsq_threshold,
            stlsq_alpha=stlsq_alpha,
            n_ensemble_models=n_ensemble_models,
            weak_seed=seed_int,
            n_weak_cells=n_weak_cells,
            weak_window_length=weak_window_length,
            p=p,
        )
        evaluation = evaluate_double_pendulum_models(
            dataset,
            fit_results,
            forecast_horizon=forecast_horizon,
            forecast_dt=forecast_dt,
            quantile_band=quantile_band,
            poly_degree=poly_degree,
            library_kind=library_kind,
            max_abs_state=max_abs_state,
        )
        for model_name, metric in evaluation.metrics.items():
            rows.append(
                {
                    "seed": seed_int,
                    "model": model_name,
                    "forecast_rmse": metric.forecast_rmse,
                    "forecast_width": metric.forecast_width,
                    "forecast_coverage": metric.forecast_coverage,
                    "forecast_valid_fraction": metric.forecast_valid_fraction,
                    "derivative_rmse": metric.derivative_rmse,
                    "derivative_width": metric.derivative_width,
                    "derivative_coverage": metric.derivative_coverage,
                    "derivative_valid_fraction": metric.derivative_valid_fraction,
                }
            )
    return rows


def summarize_fidelity_benchmark(
    benchmark_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Aggregate benchmark rows into mean/std summaries by model."""

    if not benchmark_rows:
        return []

    metric_names = [
        "forecast_rmse",
        "forecast_width",
        "forecast_coverage",
        "forecast_valid_fraction",
        "derivative_rmse",
        "derivative_width",
        "derivative_coverage",
        "derivative_valid_fraction",
    ]
    model_names = sorted({str(row["model"]) for row in benchmark_rows})
    summaries: list[dict[str, float | int | str]] = []

    for model_name in model_names:
        model_rows = [row for row in benchmark_rows if row["model"] == model_name]
        summary: dict[str, float | int | str] = {
            "model": model_name,
            "n_seeds": len({int(row["seed"]) for row in model_rows}),
        }
        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in model_rows], dtype=float)
            summary[f"{metric_name}_mean"] = float(np.nanmean(values))
            summary[f"{metric_name}_std"] = float(np.nanstd(values))
        summaries.append(summary)

    return summaries


# ---------------------------------------------------------------------------
# Part III under the shared protocol
# ---------------------------------------------------------------------------
#
# The surrogate is identified as in Part I: one HF and ten LF trajectories from
# their own initial conditions, every rung fitted at every support of the ODE
# ladder, the support chosen by the bagged spread and, per equation, the
# threshold and the polynomial degree chosen together by BIC. The library is
# polynomial and so does not contain the true (rational, trigonometric) right
# side; the surrogate is judged by forecasting held-out initial conditions.


def polynomial_library_sizes(n_states: int, max_degree: int) -> tuple[int, ...]:
    """Columns of the polynomial libraries of degree 1..max_degree, without bias.

    pysindy orders a polynomial library by degree, so the degree-d library is
    the first of these many columns of the degree-max_degree one.
    """

    return tuple(comb(n_states + d, d) - 1 for d in range(1, max_degree + 1))


@dataclass
class DoublePendulumForecastConfig(MonteCarloConfig, EnsembleConfigMixin):
    """Part III: surrogate forecasting of the double pendulum."""

    # multi-fidelity data, as in Part I
    n_hf: int = 1
    n_lf: int = 10
    noise_hf_rel: float = 0.01
    noise_lf_rel: float = 0.25

    # time discretisation; T_true is the reference trajectory the noise is scaled by.
    # The trajectories are longer than Part I's: the surrogate must cover the
    # region it forecasts in. On clean data, least squares on 11 trajectories of
    # 1 s forecasts held-out starts no better than their spread at any degree;
    # at 10 s degree 5 is within 14% (scratch check before any noisy run).
    dt: float = 1e-3
    T_train: float = 10.0
    T_true: float = 10.0

    # physical parameters (point masses on massless rods)
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81
    theta_range: tuple[float, float] = (-np.pi / 3, np.pi / 3)
    omega_range: tuple[float, float] = (-1.0, 1.0)

    # weak SINDy; BIC chooses each equation's degree in 1..poly_degree, and the
    # spread chooses the support on the degree-spread_degree library throughout
    poly_degree: int = 5
    spread_degree: int = 3
    H_xt: float | None = None
    K: int | None = None
    deduplicate: bool = True
    p: int | None = 2

    # forecasting: held-out initial conditions, over one period of the slow
    # normal mode of the linearised pendulum (2 pi / sqrt(g (2 - sqrt 2)) = 2.6 s)
    n_test: int = 5
    forecast_horizon: float = 2.6
    forecast_dt: float = 1e-2
    band: tuple[float, float] = (0.05, 0.95)
    max_abs_state: float = 50.0

    methods: tuple[str, ...] = PART1_METHODS
    results_filename: str = "double_pendulum_part3_metrics.csv"

    def make_optimizer(self) -> BICEnsembleSTLSQ:
        return BICEnsembleSTLSQ(
            n_models=self.n_ensemble_models,
            n_thresholds=self.n_thresholds,
            nested_sizes=polynomial_library_sizes(4, self.poly_degree),
            spread_size=polynomial_library_sizes(4, self.spread_degree)[-1],
        )


def _forecast_trajectories(cfg: DoublePendulumForecastConfig, n: int, seed: int, noise_abs: float):
    """``n`` trajectories over the training horizon from their own starts, with noise."""

    rng = np.random.default_rng(seed)
    noisy = []
    for _ in range(n):
        y0 = sample_double_pendulum_initial_condition(
            rng, theta_range=cfg.theta_range, omega_range=cfg.omega_range
        )
        t, X = simulate_double_pendulum_trajectory(
            y0=y0, t_final=cfg.T_train, dt=cfg.dt,
            m1=cfg.m1, m2=cfg.m2, l1=cfg.l1, l2=cfg.l2, g=cfg.g,
        )
        noisy.append(X + noise_abs * rng.standard_normal(X.shape))
    return noisy, t


def double_pendulum_reference_state_std(cfg: DoublePendulumForecastConfig) -> float:
    rng = np.random.default_rng(cfg.seed_base)
    y0 = sample_double_pendulum_initial_condition(
        rng, theta_range=cfg.theta_range, omega_range=cfg.omega_range
    )
    _, X = simulate_double_pendulum_trajectory(
        y0=y0, t_final=cfg.T_true, dt=cfg.dt,
        m1=cfg.m1, m2=cfg.m2, l1=cfg.l1, l2=cfg.l2, g=cfg.g,
    )
    return float(np.std(X))


def double_pendulum_forecast_batch(
    run_idx: int, cfg: DoublePendulumForecastConfig, noise_hf_abs: float, noise_lf_abs: float
) -> MultiTrajectoryGLSData:
    hf, t = _forecast_trajectories(cfg, cfg.n_hf, cfg.seed_base + run_idx, noise_hf_abs)
    lf, _ = _forecast_trajectories(cfg, cfg.n_lf, cfg.seed_base + 100 + run_idx, noise_lf_abs)
    return MultiTrajectoryGLSData(
        hf=hf, lf=lf, t_argument=cfg.dt,
        metadata={"t_grid": t, "weak_seed": cfg.seed_base + 10_000 + run_idx},
    )


def double_pendulum_test_cases(cfg: DoublePendulumForecastConfig, run_idx: int):
    """The run's held-out starts and the true states from them, on the forecast grid."""

    rng = np.random.default_rng(cfg.seed_base + 200_000 + run_idx)
    stride = int(round(cfg.forecast_dt / cfg.dt))
    starts, truths = [], []
    for _ in range(cfg.n_test):
        y0 = sample_double_pendulum_initial_condition(
            rng, theta_range=cfg.theta_range, omega_range=cfg.omega_range
        )
        _, X = simulate_double_pendulum_trajectory(
            y0=y0, t_final=cfg.forecast_horizon, dt=cfg.dt,
            m1=cfg.m1, m2=cfg.m2, l1=cfg.l1, l2=cfg.l2, g=cfg.g,
        )
        starts.append(y0)
        truths.append(X[::stride])
    return np.array(starts), np.array(truths)


def polynomial_rollout(coefficients: np.ndarray, y0: np.ndarray, powers: np.ndarray,
                       n_steps: int, dt: float, max_abs_state: float) -> np.ndarray:
    """RK4 rollouts of polynomial models, one per coefficient matrix (B, n_states, p).

    A rollout that leaves |x| <= max_abs_state, or stops being finite, is NaN from
    there on.
    """

    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim == 2:
        coefficients = coefficients[None]
    B = coefficients.shape[0]

    def rhs(X):
        theta = np.prod(X[:, None, :] ** powers[None], axis=2)              # (B, p)
        return np.einsum("bp,bnp->bn", theta, coefficients)

    out = np.full((B, n_steps, y0.size), np.nan)
    X = np.tile(np.asarray(y0, dtype=float), (B, 1))
    out[:, 0] = X
    alive = np.ones(B, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, n_steps):
            k1 = rhs(X)
            k2 = rhs(X + 0.5 * dt * k1)
            k3 = rhs(X + 0.5 * dt * k2)
            k4 = rhs(X + dt * k3)
            X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            alive &= np.all(np.isfinite(X), axis=1) & (np.max(np.abs(X), axis=1) <= max_abs_state)
            X = np.where(alive[:, None], X, 0.0)
            out[alive, k] = X[alive]
            if not alive.any():
                break
    return out


def forecast_metrics(model: np.ndarray, bags: np.ndarray, truth: np.ndarray,
                     band: tuple[float, float]) -> dict[str, float]:
    """Forecast scores for one held-out start.

    ``rmse`` is the error of the reported (median-coefficient) model, infinite if
    it diverges; ``width`` and ``coverage`` come from the bags that stay bounded,
    the band being their ``band`` quantiles; ``bounded`` is the share of bags
    that stay bounded over the horizon.
    """

    rmse = float(np.sqrt(np.mean((model - truth) ** 2))) if np.all(np.isfinite(model)) else np.inf
    bounded = np.all(np.isfinite(bags), axis=(1, 2))
    if not bounded.any():
        return {"rmse": rmse, "width": np.nan, "coverage": 0.0, "bounded": 0.0}
    lower, upper = np.quantile(bags[bounded], band, axis=0)
    return {
        "rmse": rmse,
        "width": float(np.mean(upper - lower)),
        "coverage": float(np.mean((truth >= lower) & (truth <= upper))),
        "bounded": float(np.mean(bounded)),
    }


def _polynomial_powers(cfg: DoublePendulumForecastConfig) -> np.ndarray:
    library = ps.PolynomialLibrary(degree=cfg.poly_degree, include_bias=False)
    return library.fit(np.zeros((1, 4))).powers_


def run_double_pendulum_forecast(cfg: DoublePendulumForecastConfig, run_idx: int, *,
                                 keep_forecasts: bool = False):
    """One Part III run: identify every rung, then forecast the held-out starts.

    Returns the long-format metric rows (averaged over the starts) and a record
    of what each rung chose; with ``keep_forecasts`` the record also holds the
    truth, the model forecast and the bag rollouts, for the figures.
    """

    from mfsindy.cases.pendulum import _pendulum_fit_multi_trajectory_weak_gls_models

    state_std = double_pendulum_reference_state_std(cfg)
    noise_hf_abs, noise_lf_abs = cfg.noise_hf_rel * state_std, cfg.noise_lf_rel * state_std
    batch = double_pendulum_forecast_batch(run_idx, cfg, noise_hf_abs, noise_lf_abs)
    methods = list(cfg.methods)
    fits = {
        support_key(variant.H_xt): _pendulum_fit_multi_trajectory_weak_gls_models(
            batch, variant, variant.make_optimizer, t_argument=batch.t_argument,
            noise_hf_abs=noise_hf_abs, noise_lf_abs=noise_lf_abs, methods=methods,
        )
        for variant in support_variants(cfg)
    }
    coefs, trace = select_rungs(fits)

    sizes = polynomial_library_sizes(4, cfg.poly_degree)
    powers = _polynomial_powers(cfg)
    starts, truths = double_pendulum_test_cases(cfg, run_idx)
    n_steps = truths.shape[1]

    rows, record = [], {"run": run_idx, "rungs": {}}
    for rung in methods:
        support = trace[rung]["support"]
        chosen = fits[support][rung].selection_["size"]
        scores, kept = [], []
        for y0, truth in zip(starts, truths):
            model = polynomial_rollout(coefs[rung][None], y0, powers, n_steps,
                                       cfg.forecast_dt, cfg.max_abs_state)[0]
            bags = polynomial_rollout(trace[rung]["bags"], y0, powers, n_steps,
                                      cfg.forecast_dt, cfg.max_abs_state)
            scores.append(forecast_metrics(model, bags, truth, cfg.band))
            if keep_forecasts:
                kept.append({"truth": truth, "model": model, "bags": bags})
        for metric in scores[0]:
            values = np.array([s[metric] for s in scores])
            value = float(np.mean(values)) if metric != "rmse" else float(np.median(values))
            rows.append({"run": run_idx, "model": rung, "metric": metric, "value": value})
        degrees = [sizes.index(int(s)) + 1 for s in chosen]
        rows.append({"run": run_idx, "model": rung, "metric": "terms",
                     "value": float(np.count_nonzero(coefs[rung]))})
        record["rungs"][rung] = {"support": support, "degree": degrees,
                                 "coefficients": coefs[rung], "forecasts": kept or None}
    return rows, record


def _forecast_run(cfg: DoublePendulumForecastConfig, run_idx: int):
    return run_double_pendulum_forecast(cfg, run_idx, keep_forecasts=run_idx == cfg.n_runs - 1)


def run_double_pendulum_forecast_experiment(cfg: DoublePendulumForecastConfig) -> pd.DataFrame:
    """The Part III Monte Carlo: metric table to ``cfg.results_filename``, and
    beside it every run's choices (support, degree per equation, coefficients),
    with the last run's forecasts."""

    from functools import partial

    from tqdm import tqdm

    from mfsindy.experiments.base import _run_one

    single_run = partial(_forecast_run, cfg)
    if cfg.n_jobs == 1:
        results = (single_run(k) for k in range(cfg.n_runs))
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=cfg.n_jobs, return_as="generator")(
            delayed(_run_one)(single_run, k, list(warnings.filters)) for k in range(cfg.n_runs)
        )
    rows, records = [], []
    for run_rows, record in tqdm(results, total=cfg.n_runs, desc="Monte Carlo double pendulum"):
        rows.extend(run_rows)
        records.append(record)

    os.makedirs(cfg.results_dir, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(cfg.results_dir, cfg.results_filename), index=False)
    with open(double_pendulum_record_path(cfg), "wb") as fh:
        pickle.dump(records, fh)
    return metrics


def double_pendulum_record_path(cfg: DoublePendulumForecastConfig) -> Path:
    return Path(cfg.results_dir) / f"{Path(cfg.results_filename).stem}_record.pkl"


__all__ = [
    "DEFAULT_HF_NOISE_SCALE",
    "DEFAULT_LF_NOISE_SCALE",
    "DEFAULT_LIBRARY_KIND",
    "DEFAULT_OMEGA_RANGE",
    "DEFAULT_POLY_DEGREE",
    "DEFAULT_THETA_RANGE",
    "DoublePendulumEvaluation",
    "DoublePendulumForecastConfig",
    "DoublePendulumForecastingDataset",
    "DoublePendulumModelMetrics",
    "DoublePendulumObservationSet",
    "EnsembleIntervalSummary",
    "WeightedWeakEnsembleFit",
    "add_omega_magnitude_noise",
    "benchmark_double_pendulum_fidelities",
    "build_double_pendulum_dataset",
    "build_double_pendulum_feature_library",
    "compute_double_pendulum_derivatives",
    "double_pendulum_forecast_batch",
    "double_pendulum_record_path",
    "double_pendulum_reference_state_std",
    "double_pendulum_test_cases",
    "forecast_metrics",
    "polynomial_library_sizes",
    "polynomial_rollout",
    "run_double_pendulum_forecast",
    "run_double_pendulum_forecast_experiment",
    "evaluate_double_pendulum_models",
    "evaluate_ensemble_sindy_derivatives",
    "fit_double_pendulum_hf_lf_models",
    "fit_double_pendulum_weighted_weak_model",
    "format_weighted_weak_equations",
    "generate_double_pendulum_trajectories",
    "make_weighted_double_pendulum_weak_library",
    "omega_variance_profile",
    "remove_fidelity_scaling_from_variance",
    "sample_double_pendulum_initial_condition",
    "simulate_double_pendulum_trajectory",
    "simulate_ensemble_sindy_forecast",
    "summarize_ensemble_forecast",
    "summarize_fidelity_benchmark",
]
