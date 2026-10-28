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
from .selection import save_selection, select_rungs, support_key, support_variants


@dataclass
class IntraTrajectoryGLSData:
    """Inputs required to fit GLS models along a single trajectory."""

    data: Any
    t_argument: Any
    libraries: Dict[str, Any]
    true_coefficients: np.ndarray


def fit_intra_trajectory_gls_model_objects(
    cfg: EnsembleConfigMixin,
    artifacts: IntraTrajectoryGLSData,
    methods: List[str],
) -> Dict[str, ps.SINDy]:
    """Fit GLS models for each weighting strategy and return the model objects."""

    models: Dict[str, ps.SINDy] = {}
    for method in methods:
        library = artifacts.libraries[method]
        model = ps.SINDy(feature_library=library, optimizer=cfg.make_optimizer())
        model.fit(artifacts.data, t=artifacts.t_argument)
        models[method] = model
    return models


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
    """Shared Monte Carlo loop for part-2 (intra-trajectory GLS) experiments.

    As in Part I, each run is fitted at every support of ``cfg.H_grid`` and each
    rung keeps the support its ensemble is tightest at, with the threshold chosen
    by BIC; the last run's choices are written beside the error table.
    """

    if not isinstance(cfg, EnsembleConfigMixin):
        raise TypeError("cfg must inherit from EnsembleConfigMixin to run GLS experiments.")

    # "Variance diag" keeps the marginal weak variances and drops the
    # correlations, so the ladder separates the two ingredients: Ones GLS is
    # correlations without variance, Variance diag is variance without
    # correlations, Variance GLS is both.
    methods = methods or getattr(cfg, "methods", None) or [
        "No weighting",
        "Ones GLS",
        "Variance diag",
        "Variance GLS",
    ]

    def error_kwargs(method: str) -> Dict[str, Any]:
        if coefficient_error_kwargs is None:
            return {}
        if callable(coefficient_error_kwargs):
            return dict(coefficient_error_kwargs(method))
        return dict(coefficient_error_kwargs)

    def single_run(run_idx: int):
        # The bagging draws from the global RNG; seeding it by run makes a run's
        # result independent of the order runs are executed in, and of the worker.
        np.random.seed((int(cfg.seed_base) + run_idx) % 2**32)
        # One fit per support of cfg.H_grid. The builder is seeded by run, so
        # every support sees the same data and only the test functions change.
        fits, true = {}, None
        for variant in support_variants(cfg):
            artifacts = run_builder(run_idx, variant)
            true = artifacts.true_coefficients
            models = fit_intra_trajectory_gls_model_objects(variant, artifacts, methods)
            fits[support_key(variant.H_xt)] = {m: model.optimizer for m, model in models.items()}
        coef_map, trace = select_rungs(fits, coef_postprocess)
        errors = {
            method: coefficient_errors(coef_map[method], true, **error_kwargs(method))
            for method in methods
        }
        last = run_idx == cfg.n_runs - 1
        return errors, {"coefs": coef_map, "true": true, "trace": trace if last else None}

    def collect(extras):
        estimates = {m: [e["coefs"][m] for e in extras] for m in methods}
        save_selection(cfg, cfg.n_runs - 1, extras[-1]["true"], extras[-1]["trace"], estimates)

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
        n_jobs=getattr(cfg, "n_jobs", 1),
        collect=collect,
    )


__all__ = [
    "IntraTrajectoryGLSData",
    "fit_intra_trajectory_gls_model_objects",
    "run_intra_trajectory_gls_experiment",
]
