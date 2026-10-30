"""Common experiment utilities shared across multi-trajectory and intra-trajectory GLS."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .selection import BICEnsembleSTLSQ

#: Seed offset for diagnostic draws (kappa reference trajectories), kept away
#: from the Monte Carlo seeds.
REFERENCE_SEED_OFFSET = 900_000


def cli_overrides(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    """Config overrides from a main script's command line: ``--runs N`` sets
    n_runs, ``--jobs N`` n_jobs. Runs are seeded by index, so the number of
    workers does not change the results."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=None, help="Monte Carlo runs (default: the config's)")
    parser.add_argument("--jobs", type=int, default=None, help="runs in parallel (default: the config's)")
    args = parser.parse_args(argv)
    overrides = {"n_runs": args.runs, "n_jobs": args.jobs}
    return {k: v for k, v in overrides.items() if v is not None}


def coefficient_errors(
    C_est: np.ndarray,
    C_true: np.ndarray,
    tol_support: float = 1e-6,
) -> tuple[float, float]:
    """Mean L1 coefficient error + normalized support mismatch for sparse coefficients."""

    C_est = np.asarray(C_est)
    C_true = np.asarray(C_true)

    if C_est.shape != C_true.shape:
        raise ValueError(
            "Shape mismatch in coefficient_errors: "
            f"C_est {C_est.shape}, C_true {C_true.shape}"
        )

    supp_true = np.abs(C_true) > tol_support
    supp_est = np.abs(C_est) > tol_support

    err = float(np.mean(np.abs(C_est - C_true)))
    n_active_terms = int(np.count_nonzero(supp_true))
    if n_active_terms == 0:
        l0_err = 0.0 if not np.any(supp_est) else float(np.count_nonzero(supp_est))
    else:
        l0_err = float(np.count_nonzero(np.logical_xor(supp_true, supp_est)) / n_active_terms)

    return err, l0_err


def _run_one(single_run_fn: Callable, run_idx: int, warning_filters: list):
    """One Monte Carlo run in a worker, with BLAS held to one thread: the runs are
    the parallel axis, and threaded BLAS inside each would oversubscribe the cores.
    The caller's warning filters are applied too, since a fresh worker process
    does not inherit them.
    """

    from threadpoolctl import threadpool_limits

    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.filters[:] = warning_filters
        return single_run_fn(run_idx)


def run_monte_carlo_experiment(
    n_runs: int,
    methods: List[str],
    single_run_fn: Callable[[int], Any],
    *,
    results_dir: str,
    results_filename: str,
    metric1_name: str,
    metric2_name: str,
    source_col: str,
    progress_desc: str,
    n_jobs: int = 1,
    collect: Callable[[List[Any]], None] | None = None,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Aggregate per-method metrics over Monte Carlo runs and persist to CSV.

    ``single_run_fn(k)`` returns the run's ``{method: (metric1, metric2)}``, or a
    pair of that and anything else the caller wants back; those extras are passed
    to ``collect`` in run order once every run is done. Runs are independent, so
    with ``n_jobs > 1`` they are spread over worker processes; each run seeds its
    own randomness, so the result does not depend on which worker ran it.
    """

    metric1_errors: Dict[str, List[float]] = {m: [] for m in methods}
    metric2_errors: Dict[str, List[float]] = {m: [] for m in methods}
    extras: List[Any] = []

    disable_progress = os.environ.get("MFSINDY_DOCS_BUILD") == "1"
    if n_jobs == 1:
        results = (single_run_fn(k) for k in range(n_runs))
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_run_one)(single_run_fn, k, list(warnings.filters)) for k in range(n_runs)
        )
    for result in tqdm(results, total=n_runs, desc=progress_desc, disable=disable_progress):
        errors, extra = result if isinstance(result, tuple) else (result, None)
        extras.append(extra)
        for m in methods:
            e1, e2 = errors[m]
            metric1_errors[m].append(e1)
            metric2_errors[m].append(e2)
    if collect is not None:
        collect(extras)

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


def paired_comparison(
    errors: Dict[str, np.ndarray], reference: str, others: List[str] | None = None
) -> pd.DataFrame:
    """Run-by-run comparison of ``reference`` against every other model.

    Per model: the median over runs of the error ratio reference / model (below 1:
    the reference is better), the share of runs the reference wins, and the
    Wilcoxon signed-rank p-value of the paired difference.
    """

    from scipy.stats import wilcoxon

    ref = np.asarray(errors[reference], dtype=float)
    rows = []
    for model in others or [m for m in errors if m != reference]:
        other = np.asarray(errors[model], dtype=float)
        rows.append({"model": model, "median ratio": float(np.median(ref / other)),
                     "reference wins": float(np.mean(ref < other)),
                     "p (Wilcoxon)": float(wilcoxon(ref, other).pvalue)})
    return pd.DataFrame(rows).set_index("model")


@dataclass
class MonteCarloConfig:
    """Common output + Monte Carlo settings shared across experiments."""

    n_runs: int = 25
    seed_base: int = 0
    n_jobs: int = 4                 # Monte Carlo runs in parallel; 1 runs them in turn
    results_dir: str = "results"
    results_filename: str = "errors.csv"


@dataclass
class EnsembleConfigMixin:
    """Ensemble settings, and the supports the estimator chooses between.

    The threshold is chosen per equation by BIC inside the ensemble, and the
    support by the spread of the bagged coefficients over ``H_grid``; see
    :mod:`mfsindy.experiments.selection`. With ``H_grid`` unset the support is
    ``H_xt`` as given.
    """

    n_ensemble_models: int = 100
    n_thresholds: int = 20
    H_grid: Optional[Sequence[Any]] = None
    #: How many test functions cover each point on average. K follows from it and
    #: the support (K = coverage * volume / support volume; extent / H for ODEs at
    #: the default), so a wider support gets proportionally fewer test functions.
    coverage: float = 2.0

    def make_optimizer(self) -> BICEnsembleSTLSQ:
        return BICEnsembleSTLSQ(n_models=self.n_ensemble_models, n_thresholds=self.n_thresholds)

    def support_grid(self) -> list:
        return list(self.H_grid) if self.H_grid is not None else [getattr(self, "H_xt", None)]


__all__ = [
    "coefficient_errors",
    "paired_comparison",
    "run_monte_carlo_experiment",
    "MonteCarloConfig",
    "EnsembleConfigMixin",
]
