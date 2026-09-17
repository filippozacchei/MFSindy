"""Helpers for multi-trajectory (Part 1) GLS experiments."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pysindy as ps

from .base import (
    EnsembleConfigMixin,
    MonteCarloConfig,
    coefficient_errors,
    run_monte_carlo_experiment,
)


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
    Restricting ``methods`` matters for per-rung tuning, where every rung is run
    under its own hyperparameters: without it each run would build and fit all
    eight rungs and discard seven.

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


def _expand_sample_weights(data: List[np.ndarray], weight: float) -> List[np.ndarray]:
    weights: List[np.ndarray] = []
    for traj in data:
        arr = np.full(traj.shape[:-1], weight, dtype=float)
        if traj.ndim >= 2:
            arr = arr[..., None]
        weights.append(arr)
    return weights


def _median_coefficients(opt: ps.EnsembleOptimizer) -> np.ndarray:
    coef_list = getattr(opt, "coef_list", None)
    if coef_list:
        arr = np.asarray(coef_list)
        if arr.ndim == 3:
            return np.median(arr, axis=0)
    return np.asarray(opt.coef_)


def fit_multi_trajectory_weak_gls_models(
    batch: MultiTrajectoryGLSData,
    optimizer_factory,
    *,
    weak_block_builder: Callable[[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray]],
    noise_hf_abs: float,
    noise_lf_abs: float,
    methods: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    """Fit the requested rungs directly in weak space by stacking per-trajectory blocks."""

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
            theta, rhs = weak_block_builder(traj, variance_field, whitener_mode=whitener_mode)
            theta_blocks.append(theta)
            rhs_blocks.append(rhs)
        return theta_blocks, rhs_blocks

    def fit_stacked(theta_blocks: list[np.ndarray], rhs_blocks: list[np.ndarray]) -> np.ndarray:
        optimizer = optimizer_factory()
        optimizer.fit(np.vstack(theta_blocks), np.vstack(rhs_blocks))
        return _median_coefficients(optimizer)

    return assemble_weak_rungs(
        build_group,
        fit_stacked,
        methods,
        seed=batch.metadata.get("weak_seed"),
    )


def fit_multi_trajectory_gls_models(
    batch: MultiTrajectoryGLSData,
    library,
    optimizer_factory: Callable[[], ps.EnsembleOptimizer],
    *,
    t_argument: Any,
    noise_hf_abs: float,
    noise_lf_abs: float,
) -> Dict[str, np.ndarray]:
    """Fit HF/LF/MF/MF_w ensemble models for a given multi-trajectory batch."""

    def make_model() -> ps.SINDy:
        return ps.SINDy(feature_library=library, optimizer=optimizer_factory())

    model_hf = make_model()
    model_lf = make_model()
    model_mf = make_model()
    model_mf_w = make_model()

    model_hf.fit(batch.hf, t=t_argument)
    print("MODEL HF:")
    model_hf.print()
    model_lf.fit(batch.lf, t=t_argument)
    print("MODEL LF:")
    model_lf.print()
    trajectories = list(batch.hf) + list(batch.lf)
    model_mf.fit(trajectories, t=t_argument)
    print("MODEL MF:")
    model_mf.print()

    eps_hf = max(float(noise_hf_abs), 1e-12)
    eps_lf = max(float(noise_lf_abs), 1e-12)
    weights = _expand_sample_weights(batch.hf, (1.0 / eps_hf) ** 2) + _expand_sample_weights(
        batch.lf, (1.0 / eps_lf) ** 2
    )
    model_mf_w.fit(trajectories, t=t_argument, sample_weight=weights)
    print("MODEL MFW:")
    model_mf_w.print()
    return {
        "HF": _median_coefficients(model_hf.optimizer),
        "LF": _median_coefficients(model_lf.optimizer),
        "MF": _median_coefficients(model_mf.optimizer),
        "MF_w": _median_coefficients(model_mf_w.optimizer),
    }


def run_multi_trajectory_gls_experiment(
    cfg: MonteCarloConfig,
    *,
    reference_state_std: Callable[[Any], float],
    dataset_builder: Callable[[int, Any, float, float], MultiTrajectoryGLSData],
    library_builder: Callable[[MultiTrajectoryGLSData, Any], Any],
    true_coefficients: Callable[[MultiTrajectoryGLSData, Any], np.ndarray],
    optimizer_factory: Callable[[], ps.EnsembleOptimizer],
    fit_models_fn: Callable[..., Dict[str, np.ndarray]] | None = None,
    coef_postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    metric1_name: str = "MAE",
    metric2_name: str = "L0",
    progress_desc: str = "Multi-trajectory GLS",
    source_col: str = "model",
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray], float, float, float]:
    """Shared Monte Carlo loop for part-1 (multi-trajectory) experiments."""

    state_std = float(reference_state_std(cfg))
    noise_hf_abs = cfg.noise_hf_rel * state_std  # type: ignore[attr-defined]
    noise_lf_abs = cfg.noise_lf_rel * state_std  # type: ignore[attr-defined]
    methods = list(getattr(cfg, "methods", None) or PART1_METHODS)

    def single_run(run_idx: int):
        batch = dataset_builder(run_idx, cfg, noise_hf_abs, noise_lf_abs)
        if fit_models_fn is None:
            library = library_builder(batch, cfg)
            coef_map = fit_multi_trajectory_gls_models(
                batch,
                library,
                optimizer_factory,
                t_argument=batch.t_argument,
                noise_hf_abs=noise_hf_abs,
                noise_lf_abs=noise_lf_abs,
            )
        else:
            coef_map = fit_models_fn(
                batch,
                cfg,
                optimizer_factory,
                t_argument=batch.t_argument,
                noise_hf_abs=noise_hf_abs,
                noise_lf_abs=noise_lf_abs,
                methods=methods,
            )
        if coef_postprocess is not None:
            coef_map = {k: coef_postprocess(v) for k, v in coef_map.items()}
        C_true = true_coefficients(batch, cfg)
        missing = [m for m in methods if m not in coef_map]
        if missing:
            raise KeyError(
                f"Fit function returned no coefficients for {missing}; "
                f"available rungs are {sorted(coef_map)}."
            )
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
    "assemble_weak_rungs",
    "fit_multi_trajectory_gls_models",
    "fit_multi_trajectory_weak_gls_models",
    "run_multi_trajectory_gls_experiment",
]
