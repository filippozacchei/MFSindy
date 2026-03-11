"""Shared utilities for case-specific experiments."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def coefficient_errors(
    C_est: np.ndarray,
    C_true: np.ndarray,
    tol_support: float = 1e-6,
    relative_to_true_support: bool = False,
) -> tuple[float, float]:
    """
    Error on coefficients and L0 (support) mismatch.

    Returns
    -------
    err : float
        Mean absolute error (restricted to the true support if requested).
    l0_err : float
        Mean support mismatch (zero vs. non-zero pattern).
    """
    C_est = np.asarray(C_est)
    C_true = np.asarray(C_true)

    if C_est.shape != C_true.shape:
        raise ValueError(
            "Shape mismatch in coefficient_errors: "
            f"C_est {C_est.shape}, C_true {C_true.shape}"
        )

    supp_true = np.abs(C_true) > tol_support
    supp_est = np.abs(C_est) > tol_support
    l0_mismatch = np.not_equal(supp_true, supp_est).astype(float)
    l0_err = float(np.mean(l0_mismatch))

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
    """
    Generic Monte Carlo loop used by multiple case studies.

    Each `single_run_fn` call returns a mapping from method name to
    a tuple (metric1, metric2). The aggregated results are saved to CSV
    and also returned as a long-form DataFrame and per-method arrays.
    """
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
        m1_arr = np.asarray(metric1_errors[m])
        m2_arr = np.asarray(metric2_errors[m])
        for run_id, (e1, e2) in enumerate(zip(m1_arr, m2_arr)):
            rows.append(
                {"run": run_id, source_col: m, "metric": metric1_name, "value": e1}
            )
            rows.append(
                {"run": run_id, source_col: m, "metric": metric2_name, "value": e2}
            )

    df_errors = pd.DataFrame(rows)
    df_errors.to_csv(errors_path, index=False)

    metric1_arrs = {m: np.asarray(metric1_errors[m]) for m in methods}
    metric2_arrs = {m: np.asarray(metric2_errors[m]) for m in methods}

    return df_errors, metric1_arrs, metric2_arrs


__all__ = ["coefficient_errors", "run_monte_carlo_experiment"]
