"""Part II pipeline for the Pendulum benchmark, as a script.

``python main2.py`` regenerates everything ``part2.ipynb`` displays: the weak
design each weighting fits, and the Monte Carlo error table. The notebook imports
these same functions, so there is one copy of the pipeline rather than two that
drift; it can either drive them or load what this script wrote.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mfsindy.cases.pendulum import (
    PendulumIntraTrajectoryGLSConfig,
    _build_pendulum_gls_artifacts,
    run_pendulum_intra_trajectory_gls_experiment,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ERRORS_PATH = RESULTS_DIR / "pendulum_part2_errors.csv"

#: The four weightings, as a 2x2 over what the covariance is allowed to carry:
#: nothing, the correlations between overlapping test functions, the per-equation
#: variances, or both. "Variance diag" is the rung that separates the last two --
#: in Part I it would collapse to a single scalar per fidelity group, because the
#: noise is constant along a trajectory there; here it does not.
METHODS = ["No weighting", "Ones GLS", "Variance diag", "Variance GLS"]

LABELS = {
    "No weighting": r"$I$",
    "Ones GLS": r"$\dot{V}\dot{V}^\top$",
    "Variance diag": r"$\mathrm{diag}(\dot{V}D_\sigma\dot{V}^\top)$",
    "Variance GLS": r"$\dot{V}D_\sigma\dot{V}^\top$",
}

METHOD_COLORS = {
    "No weighting": "tab:blue",
    "Ones GLS": "tab:orange",
    "Variance diag": "tab:brown",
    "Variance GLS": "tab:green",
}


def make_config(**overrides) -> PendulumIntraTrajectoryGLSConfig:
    settings = dict(
        n_runs=100,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    return PendulumIntraTrajectoryGLSConfig(**settings)


def weak_design(cfg, *, run_idx: int = 0) -> pd.DataFrame:
    """What each weighting actually fits.

    All four rungs share one set of test functions, so a difference between them
    is the weighting and nothing else. The diagonal spread is what makes the
    diagonal rung worth having: it is 1 for the unit-variance covariance by
    construction, and well above 1 wherever the noise varies along the trajectory.
    """

    artifacts = _build_pendulum_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base))
    rows = []
    for method in METHODS:
        library = artifacts.libraries[method]
        library.fit_transform([artifacts.data])
        row = {
            "K": int(library.K),
            "duplicates_dropped": int(getattr(library, "n_duplicate_domains_", 0)),
        }
        chol = getattr(library, "_L_chol", None)
        if chol is not None:
            diagonal = np.diag(chol @ chol.T)
            row.update(
                diag_spread=float(diagonal.max() / diagonal.min()),
                cov_cond=float(library.cov_cond_),
                cov_rank_deficit=int(library.cov_rank_deficit_),
            )
        rows.append(dict(method=method, **row))
    return pd.DataFrame(rows).set_index("method")


def run_monte_carlo(cfg, *, results_dir: Path = RESULTS_DIR,
                    verbose: bool = True) -> pd.DataFrame:
    """Run the Monte Carlo over all four weightings and write the error table."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs")
    errors, _, _ = run_pendulum_intra_trajectory_gls_experiment(cfg)
    errors.to_csv(results_dir / ERRORS_PATH.name, index=False)
    if verbose:
        print(errors.groupby(["method", "metric"])["value"].median().unstack().round(5))
    return errors


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_config()
    print(weak_design(cfg).to_string())
    print()
    run_monte_carlo(cfg)


if __name__ == "__main__":
    main()
