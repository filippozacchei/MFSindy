"""Part I pipeline for the pendulum benchmark, as a script.

``python main1.py`` regenerates everything ``part1.ipynb`` displays: the weak
design and kappa at every candidate support, the Monte Carlo error table, and
the record of how the last run chose its support and threshold. The notebook
imports these same functions, so there is one copy of the pipeline rather than
two that drift; it can either drive them or load what this script wrote.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

from mfsindy.cases.pendulum import (
    PendulumMultiTrajectoryGLSConfig,
    pendulum_kappa_by_support,
    pendulum_weak_design,
    run_pendulum_multi_trajectory_gls_experiment,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ERRORS_PATH = RESULTS_DIR / "pendulum_part1_errors.csv"

MODELS = ["HF", "LF", "MF", "VHF", "VLF", "PMF", "VMF", "MF_w"]

#: Candidate supports, as divisors of the training horizon. Every model picks
#: among them by the spread of its own ensemble, with the threshold set by BIC
#: inside the ensemble (mfsindy.experiments.selection): the same procedure for
#: every model, and no true coefficients involved.
SUPPORT_DIVISORS = (200.0, 100.0, 50.0, 20.0, 10.0, 5.0)


def make_config(**overrides) -> PendulumMultiTrajectoryGLSConfig:
    settings = dict(
        p=2,
        n_runs=100,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    cfg = PendulumMultiTrajectoryGLSConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = [cfg.T_train / d for d in SUPPORT_DIVISORS]
    return cfg


def design_report(cfg) -> pd.DataFrame:
    """The weak design at every candidate support, with kappa there.

    Kappa is the median over clean reference trajectories, as in
    :func:`pendulum_kappa_by_support`; it is defined at the true coefficients,
    so it is reported, not used to choose anything.
    """

    kappa = pendulum_kappa_by_support(cfg, cfg.H_grid)
    design = pd.DataFrame([pendulum_weak_design(replace(cfg, H_xt=H, K=None)) for H in cfg.H_grid])
    return pd.concat([kappa.reset_index(drop=True), design.drop(columns="K", errors="ignore")], axis=1)


def run_monte_carlo(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Run the Monte Carlo; every model selects its own support and threshold.

    The runner writes the error table to ``cfg.results_dir``, and beside it the
    last run's selection (``mfsindy.experiments.load_selection``).
    """

    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs over {len(cfg.H_grid)} supports")
    errors = run_pendulum_multi_trajectory_gls_experiment(cfg)[0]
    if verbose:
        print(errors.groupby(["model", "metric"])["value"].median().unstack().round(4))
    return errors


def main() -> None:
    cfg = make_config()
    print(design_report(cfg).to_string(index=False))
    run_monte_carlo(cfg)


if __name__ == "__main__":
    main()
