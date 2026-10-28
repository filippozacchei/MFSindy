"""Helpers for multi-trajectory (Part 1) GLS experiments."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from .base import (
    MonteCarloConfig,
    coefficient_errors,
    run_monte_carlo_experiment,
)
from .selection import save_selection, select_rungs, support_key, support_variants


#: Part I rungs, in reporting order. PMF and VMF are the baselines required by
#: review: the weak-SINDy covariance applied blind to fidelity, and per-group
#: weighting by the marginal weak variances alone. A case config may override
#: this by defining a ``methods`` field.
PART1_METHODS = ("HF", "LF", "MF", "VHF", "VLF", "PMF", "VMF", "MF_w")

#: Which weak blocks each rung is built from, as (fidelity, weighting) pairs.
#: The weightings are: ``plain`` (no whitening), ``weighted`` (the fidelity's own
#: noise level, full weak covariance), ``pooled`` (unit variance, so only the
#: test-function correlations survive) and ``diag`` (the fidelity's noise level
#: with the marginal weak variances only).
RUNG_BLOCKS: Dict[str, tuple[tuple[str, str], ...]] = {
    "HF": (("hf", "plain"),),
    "LF": (("lf", "plain"),),
    "MF": (("hf", "plain"), ("lf", "plain")),
    "VHF": (("hf", "weighted"),),
    "VLF": (("lf", "weighted"),),
    "PMF": (("hf", "pooled"), ("lf", "pooled")),
    "VMF": (("hf", "diag"), ("lf", "diag")),
    "MF_w": (("hf", "weighted"), ("lf", "weighted")),
}


#: Rungs that whiten by the weak covariance, and so depend on the covariance
#: model being valid. Kappa bounds the support for these and says nothing about
#: the rest: HF, LF and MF form no covariance at all. Derived from the recipes
#: above so it cannot drift from them.
WHITENING_RUNGS: frozenset[str] = frozenset(
    rung
    for rung, blocks in RUNG_BLOCKS.items()
    if any(weighting != "plain" for _, weighting in blocks)
)


#: Rungs whitened by the FULL weak covariance, which is what kappa bounds.
#: VMF is deliberately not here: it whitens by the marginal variances alone, and
#: GLS is invariant to a global rescale of its weights, so the level of kappa
#: cancels for it -- at a kappa of 3.2 its weights are off by only 1.5x. The
#: rungs below use the test-function correlations, where the error does not
#: cancel, and PMF uses nothing else.
FULL_COVARIANCE_RUNGS: frozenset[str] = frozenset(
    rung
    for rung, blocks in RUNG_BLOCKS.items()
    if any(weighting in ("weighted", "pooled") for _, weighting in blocks)
)


def assemble_weak_rungs(
    group_builder: Callable[[str, str], tuple[List[np.ndarray], List[np.ndarray]]],
    fit_stacked: Callable[[List[np.ndarray], List[np.ndarray]], np.ndarray],
    methods: Sequence[str] | None = None,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Build and fit only the rungs in ``methods``, sharing blocks between them.

    ``group_builder(fidelity, weighting)`` returns the weak (theta, rhs) blocks
    for one group; each distinct pair is built at most once, so rungs that share
    a group (MF_w and VHF both use the weighted HF blocks) pay for it once.
    Restricting ``methods`` lets a run fit a subset without building the rest.

    ``seed`` fixes the bootstrap draws of the ensemble fit, one stream per rung.
    The ensemble optimizer bags from the global RNG, so without this a rung's
    draws depend on how many rungs happened to be fitted before it: the same run
    would give different coefficients depending on the subset requested, and
    would not repeat across sessions. Seeding happens after the blocks are built,
    since building a weak library reseeds the global RNG itself to keep the test
    functions in a common position across rungs.
    """

    wanted = list(methods or PART1_METHODS)
    unknown = [m for m in wanted if m not in RUNG_BLOCKS]
    if unknown:
        raise KeyError(
            f"No block recipe for {unknown}; known rungs are {sorted(RUNG_BLOCKS)}."
        )

    cache: Dict[tuple[str, str], tuple[List[np.ndarray], List[np.ndarray]]] = {}

    def group(key: tuple[str, str]):
        if key not in cache:
            cache[key] = group_builder(*key)
        return cache[key]

    coefficients: Dict[str, np.ndarray] = {}
    for rung in wanted:
        theta_blocks: List[np.ndarray] = []
        rhs_blocks: List[np.ndarray] = []
        for key in RUNG_BLOCKS[rung]:
            theta, rhs = group(key)
            theta_blocks = theta_blocks + list(theta)
            rhs_blocks = rhs_blocks + list(rhs)
        if seed is not None:
            np.random.seed((int(seed) + zlib.crc32(rung.encode())) % 2**32)
        coefficients[rung] = fit_stacked(theta_blocks, rhs_blocks)
    return coefficients


@dataclass
class MultiTrajectoryGLSData:
    """Inputs required for a single multi-trajectory GLS run."""

    hf: List[np.ndarray]
    lf: List[np.ndarray]
    t_argument: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


def _variance_signature(variance_field: np.ndarray | None):
    """A key for the library a variance field produces, or None if unshareable.

    Part I weights every sample of a group alike -- ``np.full(shape, sigma**2)``
    -- so one library serves the whole group. A field that varies sample to
    sample would need its own whitener, and returning None there keeps this from
    silently handing back the wrong one.
    """

    if variance_field is None:
        return ("plain",)
    array = np.asarray(variance_field)
    low, high = float(array.min()), float(array.max())
    if low != high:
        return None
    return (array.shape, low)


def fit_multi_trajectory_weak_gls_models(
    batch: MultiTrajectoryGLSData,
    optimizer_factory,
    *,
    weak_library_builder: Callable[..., Any],
    noise_hf_abs: float,
    noise_lf_abs: float,
    methods: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Fit the requested rungs directly in weak space by stacking per-trajectory blocks.

    Returns the fitted optimizer per rung, so the caller can read what the
    ensemble chose as well as the coefficients.

    ``weak_library_builder(variance_field, whitener_mode=...)`` returns the weak
    library for one group. It is called once per group rather than once per
    trajectory: the domain placement, the quadrature weights and the covariance
    Cholesky depend on the grid, the seed, the support and the variance field,
    all of which a group holds fixed -- only the transform depends on the data.
    Rebuilding per trajectory repeated that work 11 times over for a 10-LF
    group.

    Reusing one library per group also enforces what the shared ``weak_seed``
    already intends: every trajectory in a group is integrated against the same
    test functions. Note the library is fitted once and then only transformed --
    calling ``fit`` again would redraw the domains from the global RNG without
    reseeding, quietly changing the test functions mid-group.
    """

    libraries: Dict[Any, Any] = {}

    def library_for(variance_field: np.ndarray | None, whitener_mode: str, sample: np.ndarray):
        signature = _variance_signature(variance_field)
        key = None if signature is None else (signature, whitener_mode)
        if key is not None and key in libraries:
            return libraries[key]
        library = weak_library_builder(variance_field, whitener_mode=whitener_mode)
        # Fit once: the geometry does not depend on which trajectory is passed.
        library.fit([sample])
        if key is not None:
            libraries[key] = library
        return library

    def build_group(fidelity: str, weighting: str):
        trajectories = batch.hf if fidelity == "hf" else batch.lf
        noise_abs = noise_hf_abs if fidelity == "hf" else noise_lf_abs
        # PMF is blind to fidelity: a variance common to every trajectory cancels
        # in the least-squares solution, so only the correlation structure of the
        # test functions is retained. VMF keeps the fidelity's own noise level but
        # discards those correlations, weighting by the marginal weak variances.
        if weighting == "pooled":
            noise_abs = 1.0
        whitener_mode = "diag" if weighting == "diag" else "full"

        theta_blocks: list[np.ndarray] = []
        rhs_blocks: list[np.ndarray] = []
        for traj in trajectories:
            variance_field = (
                None
                if weighting == "plain"
                else np.full(traj.shape[:-1], noise_abs**2, dtype=float)
            )
            library = library_for(variance_field, whitener_mode, traj)
            theta_blocks.append(np.asarray(library.transform([traj])[0]))
            rhs_blocks.append(np.asarray(library.convert_u_dot_integral(traj)))
        return theta_blocks, rhs_blocks

    def fit_stacked(theta_blocks: list[np.ndarray], rhs_blocks: list[np.ndarray]):
        optimizer = optimizer_factory()
        optimizer.fit(np.vstack(theta_blocks), np.vstack(rhs_blocks))
        return optimizer

    return assemble_weak_rungs(
        build_group,
        fit_stacked,
        methods,
        seed=batch.metadata.get("weak_seed"),
    )


def run_multi_trajectory_gls_experiment(
    cfg: MonteCarloConfig,
    *,
    reference_state_std: Callable[[Any], float],
    dataset_builder: Callable[[int, Any, float, float], MultiTrajectoryGLSData],
    true_coefficients: Callable[[MultiTrajectoryGLSData, Any], np.ndarray],
    optimizer_factory: Callable[[], Any],
    fit_models_fn: Callable[..., Dict[str, Any]],
    coef_postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    metric1_name: str = "MAE",
    metric2_name: str = "L0",
    progress_desc: str = "Multi-trajectory GLS",
    source_col: str = "model",
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray], float, float, float]:
    """Shared Monte Carlo loop for part-1 (multi-trajectory) experiments.

    Each run is fitted at every support of ``cfg.H_grid``; each rung keeps the
    support its ensemble is tightest at, with the threshold chosen by BIC inside
    the ensemble (:mod:`mfsindy.experiments.selection`). The last run's choices
    are written beside the error table for plotting.
    """

    state_std = float(reference_state_std(cfg))
    noise_hf_abs = cfg.noise_hf_rel * state_std  # type: ignore[attr-defined]
    noise_lf_abs = cfg.noise_lf_rel * state_std  # type: ignore[attr-defined]
    methods = list(getattr(cfg, "methods", None) or PART1_METHODS)

    # Every run's reported coefficients, kept for the across-runs plot.
    estimates: Dict[str, list] = {m: [] for m in methods}

    def single_run(run_idx: int):
        batch = dataset_builder(run_idx, cfg, noise_hf_abs, noise_lf_abs)
        fits = {
            support_key(variant.H_xt): fit_models_fn(
                batch,
                variant,
                optimizer_factory,
                t_argument=batch.t_argument,
                noise_hf_abs=noise_hf_abs,
                noise_lf_abs=noise_lf_abs,
                methods=methods,
            )
            for variant in support_variants(cfg)
        }
        coef_map, trace = select_rungs(
            fits, None if coef_postprocess is None else (lambda coef, _rung: coef_postprocess(coef))
        )
        C_true = true_coefficients(batch, cfg)
        for m in methods:
            estimates[m].append(coef_map[m])
        if run_idx == cfg.n_runs - 1:
            save_selection(cfg, run_idx, C_true, trace, estimates)
        return {
            method: coefficient_errors(coef_map[method], C_true)
            for method in methods
        }

    df_errors, metric1, metric2 = run_monte_carlo_experiment(
        n_runs=cfg.n_runs,
        methods=methods,
        single_run_fn=single_run,
        results_dir=cfg.results_dir,
        results_filename=cfg.results_filename,
        metric1_name=metric1_name,
        metric2_name=metric2_name,
        source_col=source_col,
        progress_desc=progress_desc,
    )

    return df_errors, metric1, metric2, state_std, noise_hf_abs, noise_lf_abs


__all__ = [
    "MultiTrajectoryGLSData",
    "RUNG_BLOCKS",
    "WHITENING_RUNGS",
    "FULL_COVARIANCE_RUNGS",
    "assemble_weak_rungs",
    "fit_multi_trajectory_weak_gls_models",
    "run_multi_trajectory_gls_experiment",
]
