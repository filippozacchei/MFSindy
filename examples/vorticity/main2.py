"""Part II pipeline for the vorticity-flow benchmark, as a script.

``python main2.py`` runs the Monte Carlo of the four weightings and writes the
error table, as for the isothermal flow.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mfsindy.experiments import cli_overrides
from mfsindy.cases.vorticity_flow import (
    VorticityIntraTrajectoryGLSConfig,
    run_vorticity_intra_trajectory_gls_experiment,
    vorticity_run_trajectories,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ERRORS_PATH = RESULTS_DIR / "vorticity_part2_errors.csv"

METHODS = ["No weighting", "Ones GLS", "Variance diag", "Variance GLS"]
SPACE_DIVISORS = (8.0, 4.0)
TIME_DIVISORS = (100.0, 50.0, 20.0, 10.0, 5.0)


def supports(cfg) -> list:
    return [[cfg.window / ds, cfg.window / ds, cfg.T / dt] for ds in SPACE_DIVISORS for dt in TIME_DIVISORS]


def make_config(**overrides) -> VorticityIntraTrajectoryGLSConfig:
    settings = dict(n_runs=100, results_dir=str(RESULTS_DIR), results_filename=ERRORS_PATH.name)
    settings.update(overrides)
    cfg = VorticityIntraTrajectoryGLSConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = supports(cfg)
    return cfg


def run_trajectories(cfg, run_idx: int = 0, *, observed: dict | None = None):
    """Run run_idx's windows as the Monte Carlo draws them: ``(t, grid, clean, noisy, variance)``."""

    return vorticity_run_trajectories(cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx), observed=observed)


def run_monte_carlo(cfg, *, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs")
    errors, _, _ = run_vorticity_intra_trajectory_gls_experiment(cfg)
    if verbose:
        print(errors.groupby(["method", "metric"])["value"].median().unstack().round(6))
    return errors


def main() -> None:
    run_monte_carlo(make_config(**cli_overrides()))


if __name__ == "__main__":
    main()
