"""Weak-system builders shared by the Part II cases."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pysindy as ps

from mfsindy.weighted_weak_pde_library import (
    DedupedWeakPDELibrary,
    WeightedWeakPDELibrary,
)

from .intra_trajectory import IntraTrajectoryGLSData


def build_polynomial_intra_trajectory_artifacts(
    data: np.ndarray,
    t_grid: np.ndarray,
    *,
    variance: np.ndarray,
    poly_degree: int,
    derivative_order: int = 1,
    include_bias: bool = False,
    K: int | None = None,
    H_xt: float | None = None,
    p: int | None = None,
    weak_seed: int | None = None,
) -> IntraTrajectoryGLSData:
    """Construct weak / weighted-weak libraries for one ODE trajectory segment."""

    data = np.asarray(data, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    variance = np.asarray(variance, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must have shape (n_steps, n_states).")
    if variance.shape not in ((data.shape[0],), data.shape):
        raise ValueError(
            f"variance must have shape ({data.shape[0]},), or {data.shape} for one variance "
            f"per component, got {variance.shape}."
        )
    if t_grid.shape != (data.shape[0],):
        raise ValueError(f"t_grid must have shape ({data.shape[0]},), got {t_grid.shape}.")

    XT = t_grid[:, None]
    base_library = ps.PolynomialLibrary(
        degree=poly_degree,
        include_bias=include_bias,
    )

    if weak_seed is not None:
        np.random.seed(int(weak_seed))

    common_kwargs = {
        "function_library": base_library,
        "derivative_order": derivative_order,
        "spatiotemporal_grid": XT,
        "is_uniform": True,
        "include_bias": include_bias,
    }
    if K is not None:
        common_kwargs["K"] = K
    if H_xt is not None:
        common_kwargs["H_xt"] = H_xt
    if p is not None:
        common_kwargs["p"] = p

    weak_lib = DedupedWeakPDELibrary(**common_kwargs)

    if weak_seed is not None:
        np.random.seed(int(weak_seed))
    weighted_weak_lib_var = WeightedWeakPDELibrary(
        spatiotemporal_weights=variance,
        **common_kwargs,
    )

    if weak_seed is not None:
        np.random.seed(int(weak_seed))
    weighted_weak_lib_ones = WeightedWeakPDELibrary(
        spatiotemporal_weights=np.ones(data.shape[0]),        # one unit variance, shared
        **common_kwargs,
    )

    if weak_seed is not None:
        np.random.seed(int(weak_seed))
    weighted_weak_lib_var_diag = WeightedWeakPDELibrary(
        spatiotemporal_weights=variance,
        **common_kwargs,
        whitener_mode="diag",
    )
    return IntraTrajectoryGLSData(
        data=data,
        t_argument=t_grid,
        libraries={
            "No weighting": weak_lib,
            "Variance GLS": weighted_weak_lib_var,
            "Ones GLS": weighted_weak_lib_ones,
            "Variance diag": weighted_weak_lib_var_diag,
        },
        true_coefficients=np.empty((0, 0)),
    )


def build_ode_run_artifacts(
    cfg,
    trajectories,
    t_grid: np.ndarray,
    variances,
    *,
    weak_seed: int,
    true_coefficients: np.ndarray,
) -> list[IntraTrajectoryGLSData]:
    """The four weightings for each trajectory of one Part II ODE run.

    K follows the support at ``cfg.coverage``, as in Part I. Every trajectory and
    every rung is placed on the same test functions (one seed), so the rungs
    differ by covariance model alone. The variances are passed as they are, not
    rescaled per trajectory: the trajectories' weak systems are stacked, and a
    per-trajectory rescaling would change their relative weights.
    """

    extent = float(np.ptp(t_grid))
    H = cfg.H_xt if cfg.H_xt is not None else extent / 20.0
    K = int(cfg.K) if cfg.K is not None else max(2, int(round(cfg.coverage / 2 * extent / H)))
    out = []
    for data, variance in zip(trajectories, variances):
        artifacts = build_polynomial_intra_trajectory_artifacts(
            data, t_grid, variance=variance, poly_degree=cfg.poly_degree,
            derivative_order=cfg.derivative_order, include_bias=cfg.include_bias,
            K=K, H_xt=H, p=cfg.p, weak_seed=weak_seed,
        )
        artifacts.true_coefficients = true_coefficients
        out.append(artifacts)
    return out


def pde_coverage_test_function_count(spatiotemporal_grid, H_xt, coverage: float = 2.0) -> int:
    """K at the given coverage (2: every point under two supports) on a grid.

    The ODE cases resolve K as ``extent / H``; in more than one dimension the
    same statement is a ratio of volumes, ``2 * prod(extents) / prod(2H)``. It
    is written once here because Part I's five case builders each carried a copy
    and Part II carried none -- the PDE cases asked for a fixed 100 (the flow for
    1000) whatever the support, which is a different coverage at every candidate
    and so not a comparison.
    """

    grid = np.asarray(spatiotemporal_grid, dtype=float)
    extents = [float(np.ptp(grid[..., axis])) for axis in range(grid.shape[-1])]
    widths = np.atleast_1d(np.asarray(H_xt, dtype=float))
    if widths.size == 1:
        widths = np.full(len(extents), float(widths[0]))
    domain = float(np.prod(2.0 * widths))
    if domain <= 0.0:
        raise ValueError(f"support widths must be positive, got {H_xt!r}.")
    return max(2, int(round(coverage * float(np.prod(extents)) / domain)))


def build_pde_intra_trajectory_artifacts(
    data: np.ndarray,
    spatiotemporal_grid: np.ndarray,
    *,
    t_grid: np.ndarray,
    variance: np.ndarray,
    derivative_order: int,
    poly_degree: int | None = None,
    function_library=None,
    include_bias: bool = False,
    function_include_bias: bool | None = None,
    K: int | None = None,
    H_xt=None,
    coverage: float = 2.0,
    p: int | None = None,
    weak_seed: int | None = None,
    true_coefficients: np.ndarray | None = None,
) -> IntraTrajectoryGLSData:
    """The four Part II weightings for one noisy PDE field.

    The PDE counterpart of :func:`build_polynomial_intra_trajectory_artifacts`,
    and it exists for the same reason: the search has to build libraries on a
    slice of the data, which the case builder cannot do because it generates its
    own. Every rung is placed on the same test functions, so the comparison is
    between covariance models and not between draws.

    ``K`` follows the support at ``coverage`` when left out.

    The feature library is either given as ``function_library`` -- the flow case
    builds a CustomLibrary of its own terms -- or constructed as a polynomial
    library of degree ``poly_degree``; exactly one of the two is required.

    ``include_bias`` adds the constant term to the weak feature set;
    ``function_include_bias`` says whether the inner polynomial library carries
    one too, and defaults to the same. Burgers sets it False while the weak
    library keeps the bias, because otherwise the constant appears twice and the
    duplicate column makes Theta rank-deficient.
    """

    data = np.asarray(data, dtype=float)
    grid = np.asarray(spatiotemporal_grid, dtype=float)
    variance = np.asarray(variance, dtype=float)

    if variance.shape != data.shape[:-1]:
        raise ValueError(
            f"variance must match the field's grid shape {data.shape[:-1]}, "
            f"got {variance.shape}."
        )
    if grid.shape[:-1] != data.shape[:-1]:
        raise ValueError(
            f"spatiotemporal_grid covers {grid.shape[:-1]} but the field is "
            f"{data.shape[:-1]}; the two must agree point for point."
        )

    if (function_library is None) == (poly_degree is None):
        raise ValueError("pass exactly one of function_library or poly_degree.")
    base_library = function_library
    if base_library is None:
        base_library = ps.PolynomialLibrary(
            degree=poly_degree,
            include_bias=(
                include_bias if function_include_bias is None else function_include_bias
            ),
        )

    common_kwargs = {
        "function_library": base_library,
        "derivative_order": derivative_order,
        "spatiotemporal_grid": grid,
        "is_uniform": True,
        "include_bias": include_bias,
    }
    if H_xt is not None:
        common_kwargs["H_xt"] = H_xt
    if K is not None:
        common_kwargs["K"] = int(K)
    elif H_xt is not None:
        common_kwargs["K"] = pde_coverage_test_function_count(grid, H_xt, coverage)
    # with neither, the library picks both, which is only sensible for a smoke test
    if p is not None:
        common_kwargs["p"] = p

    def seeded():
        if weak_seed is not None:
            np.random.seed(int(weak_seed))

    seeded()
    weak_lib = DedupedWeakPDELibrary(**common_kwargs)
    seeded()
    weighted_var = WeightedWeakPDELibrary(spatiotemporal_weights=variance, **common_kwargs)
    seeded()
    weighted_ones = WeightedWeakPDELibrary(
        spatiotemporal_weights=np.ones_like(variance), **common_kwargs
    )
    seeded()
    weighted_var_diag = WeightedWeakPDELibrary(
        spatiotemporal_weights=variance, **common_kwargs, whitener_mode="diag"
    )

    return IntraTrajectoryGLSData(
        data=data,
        t_argument=np.asarray(t_grid, dtype=float),
        libraries={
            "No weighting": weak_lib,
            "Variance GLS": weighted_var,
            "Ones GLS": weighted_ones,
            "Variance diag": weighted_var_diag,
        },
        true_coefficients=(
            np.empty((0, 0)) if true_coefficients is None else np.asarray(true_coefficients)
        ),
    )


def get_library_feature_names(
    library: Any,
    reference_data: np.ndarray,
    *,
    input_features: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Best-effort feature-name extraction for a fitted SINDy library."""

    try:
        library.fit([reference_data])
    except Exception:
        try:
            library.fit(reference_data)
        except Exception:
            return ()

    getter = getattr(library, "get_feature_names", None)
    if getter is None:
        return ()

    for kwargs in (
        {"input_features": list(input_features)} if input_features is not None else None,
        {},
    ):
        if kwargs is None:
            continue
        try:
            names = getter(**kwargs)
            return tuple(str(name) for name in names)
        except Exception:
            continue

    try:
        names = getter(list(input_features)) if input_features is not None else getter()
        return tuple(str(name) for name in names)
    except Exception:
        return ()


__all__ = [
    "build_pde_intra_trajectory_artifacts",
    "build_polynomial_intra_trajectory_artifacts",
    "get_library_feature_names",
    "pde_coverage_test_function_count",
]
