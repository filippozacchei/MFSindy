"""Helpers for intra-trajectory (Part 2) GLS experiments."""

from __future__ import annotations

from dataclasses import dataclass
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
class IntraTrajectoryGLSData:
    """Inputs required to fit GLS models along a single trajectory."""

    data: Any
    t_argument: Any
    libraries: Dict[str, Any]
    true_coefficients: np.ndarray


def fit_intra_trajectory_gls_models(
    cfg: EnsembleConfigMixin,
    artifacts: IntraTrajectoryGLSData,
    methods: List[str],
    *,
    coef_postprocess: Callable[[np.ndarray, str], np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    """Fit GLS models for each weighting strategy."""

    coefs: Dict[str, np.ndarray] = {}
    for method in methods:
        library = artifacts.libraries[method]
        model = ps.SINDy(feature_library=library, optimizer=cfg.make_optimizer())
        model.fit(artifacts.data, t=artifacts.t_argument)
        coef = np.asarray(model.optimizer.coef_)
        if coef_postprocess is not None:
            coef = coef_postprocess(coef, method)
        coefs[method] = coef
    return coefs


def run_intra_trajectory_gls_experiment(
    cfg: MonteCarloConfig,
    *,
    run_builder: Callable[[int, Any], IntraTrajectoryGLSData],
    progress_desc: str,
    methods: List[str] | None = None,
    metric1_name: str = "L1",
    metric2_name: str = "L0",
    source_col: str = "method",
    coef_postprocess: Callable[[np.ndarray, str], np.ndarray] | None = None,
    coefficient_error_kwargs: Callable[[str], Dict[str, Any]] | Dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Shared Monte Carlo loop for part-2 (intra-trajectory GLS) experiments."""

    if not isinstance(cfg, EnsembleConfigMixin):
        raise TypeError("cfg must inherit from EnsembleConfigMixin to run GLS experiments.")

    methods = methods or ["No weighting", "Variance GLS", "Ones GLS"]

    def error_kwargs(method: str) -> Dict[str, Any]:
        if coefficient_error_kwargs is None:
            return {}
        if callable(coefficient_error_kwargs):
            return dict(coefficient_error_kwargs(method))
        return dict(coefficient_error_kwargs)

    def single_run(run_idx: int):
        artifacts = run_builder(run_idx, cfg)
        coef_map = fit_intra_trajectory_gls_models(
            cfg,
            artifacts,
            methods,
            coef_postprocess=coef_postprocess,
        )
        return {
            method: coefficient_errors(
                coef_map[method],
                artifacts.true_coefficients,
                **error_kwargs(method),
            )
            for method in methods
        }

    return run_monte_carlo_experiment(
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


__all__ = [
    "IntraTrajectoryGLSData",
    "fit_intra_trajectory_gls_models",
    "run_intra_trajectory_gls_experiment",
]
