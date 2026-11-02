"""Part I pipeline for the vorticity-flow benchmark, as a script.

``python main1.py`` runs the Monte Carlo and writes the error table and the
record of how the last run chose its support and threshold, as for the
isothermal flow.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from mfsindy.experiments import cli_overrides
from mfsindy.cases.vorticity_flow import (
    VorticityMultiTrajectoryGLSConfig,
    run_vorticity_multi_trajectory_gls_experiment,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ERRORS_PATH = RESULTS_DIR / "vorticity_part1_errors.csv"

#: Candidate supports, as for the isothermal flow and Burgers: in space W/8 and
#: W/4 of the window edge W, in time the ODE divisors of the horizon without
#: T/200. Every model picks among them by the spread of its own ensemble, with
#: the threshold set by BIC inside the ensemble.
SPACE_DIVISORS = (8.0, 4.0)
TIME_DIVISORS = (100.0, 50.0, 20.0, 10.0, 5.0)


def make_config(**overrides) -> VorticityMultiTrajectoryGLSConfig:
    settings = dict(n_runs=100, n_lf=10, n_hf=1, p=2, results_dir=str(RESULTS_DIR),
                    results_filename=ERRORS_PATH.name)
    settings.update(overrides)
    cfg = VorticityMultiTrajectoryGLSConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = [[cfg.window / ds, cfg.window / ds, cfg.T / dt]
                      for ds in SPACE_DIVISORS for dt in TIME_DIVISORS]
    return cfg


def run_monte_carlo(cfg, *, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs over {len(cfg.H_grid)} supports")
    errors = run_vorticity_multi_trajectory_gls_experiment(cfg)[0]
    if verbose:
        print(errors.groupby(["model", "metric"])["value"].median().unstack().round(6))
    return errors


def main() -> None:
    run_monte_carlo(make_config(**cli_overrides()))


if __name__ == "__main__":
    main()
