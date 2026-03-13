"""Helpers for multi-trajectory (Part 1) GLS experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
import pysindy as ps

from .base import (
    EnsembleConfigMixin,
    MonteCarloConfig,
    coefficient_errors,
    run_monte_carlo_experiment,
)


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
    methods = ["HF", "LF", "MF", "MF_w"]

    def single_run(run_idx: int):
        batch = dataset_builder(run_idx, cfg, noise_hf_abs, noise_lf_abs)
        library = library_builder(batch, cfg)
        coef_map = fit_multi_trajectory_gls_models(
            batch,
            library,
            optimizer_factory,
            t_argument=batch.t_argument,
            noise_hf_abs=noise_hf_abs,
            noise_lf_abs=noise_lf_abs,
        )
        if coef_postprocess is not None:
            coef_map = {k: coef_postprocess(v) for k, v in coef_map.items()}
        C_true = true_coefficients(batch, cfg)
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
    "fit_multi_trajectory_gls_models",
    "run_multi_trajectory_gls_experiment",
]
