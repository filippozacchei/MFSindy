"""Common experiment utilities shared across multi-trajectory and intra-trajectory GLS."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pysindy as ps
from tqdm import tqdm


def coefficient_errors(
    C_est: np.ndarray,
    C_true: np.ndarray,
    tol_support: float = 1e-6,
    relative_to_true_support: bool = False,
) -> tuple[float, float]:
    """Mean absolute error + support mismatch for sparse coefficient matrices."""

    C_est = np.asarray(C_est)
    C_true = np.asarray(C_true)

    if C_est.shape != C_true.shape:
        raise ValueError(
            "Shape mismatch in coefficient_errors: "
            f"C_est {C_est.shape}, C_true {C_true.shape}"
        )

    supp_true = np.abs(C_true) > tol_support
    supp_est = np.abs(C_est) > tol_support
    l0_err = float(np.mean(np.not_equal(supp_true, supp_est)))

    if relative_to_true_support and np.any(supp_true):
        err = float(np.mean(np.abs(C_est[supp_true] - C_true[supp_true])))
    else:
        err = float(np.mean(np.abs(C_est - C_true)))

    return err, l0_err


def run_monte_carlo_experiment(
    n_runs: int,
    methods: List[str],
    single_run_fn: Callable[[int], Dict[str, Tuple[float, float]]],
    *,
    results_dir: str,
    results_filename: str,
    metric1_name: str,
    metric2_name: str,
    source_col: str,
    progress_desc: str,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Aggregate per-method metrics over Monte Carlo runs and persist to CSV."""

    metric1_errors: Dict[str, List[float]] = {m: [] for m in methods}
    metric2_errors: Dict[str, List[float]] = {m: [] for m in methods}

    for k in tqdm(range(n_runs), desc=progress_desc):
        errors = single_run_fn(k)
        for m in methods:
            e1, e2 = errors[m]
            metric1_errors[m].append(e1)
            metric2_errors[m].append(e2)

    os.makedirs(results_dir, exist_ok=True)
    errors_path = os.path.join(results_dir, results_filename)

    rows = []
    for m in methods:
        for run_id, (e1, e2) in enumerate(zip(metric1_errors[m], metric2_errors[m])):
            rows.append({"run": run_id, source_col: m, "metric": metric1_name, "value": e1})
            rows.append({"run": run_id, source_col: m, "metric": metric2_name, "value": e2})

    df_errors = pd.DataFrame(rows)
    df_errors.to_csv(errors_path, index=False)

    metric1_arrs = {m: np.asarray(vals) for m, vals in metric1_errors.items()}
    metric2_arrs = {m: np.asarray(vals) for m, vals in metric2_errors.items()}

    return df_errors, metric1_arrs, metric2_arrs


@dataclass
class MonteCarloConfig:
    """Common output + Monte Carlo settings shared across experiments."""

    n_runs: int = 25
    seed_base: int = 0
    results_dir: str = "results"
    results_filename: str = "errors.csv"


@dataclass
class EnsembleConfigMixin:
    """Mixin providing SINDy ensemble hyperparameters."""

    stlsq_threshold: float = 0.5
    n_ensemble_models: int = 100

    def stlsq_kwargs(self) -> Dict[str, Any]:
        return {}

    def ensemble_kwargs(self) -> Dict[str, Any]:
        return {"bagging": True}

    def make_optimizer(self) -> ps.EnsembleOptimizer:
        base_opt = ps.STLSQ(self.stlsq_threshold, **self.stlsq_kwargs())
        return ps.EnsembleOptimizer(
            base_opt,
            n_models=self.n_ensemble_models,
            **self.ensemble_kwargs(),
        )


__all__ = [
    "coefficient_errors",
    "run_monte_carlo_experiment",
    "MonteCarloConfig",
    "EnsembleConfigMixin",
]
