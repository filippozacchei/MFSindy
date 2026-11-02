"""Part III pipeline for the double-pendulum forecasting benchmark, as a script.

``python main3.py`` identifies a polynomial surrogate for every Part I model and
forecasts the observed trajectory 10 s past the end of training with it,
scoring the states and their derivatives (RMSE, R^2, band width and coverage)
over the first 2.5, 5 and 10 s, and how long each forecast stays valid. One
trajectory is observed by one HF and ten LF sensors (1% and 25% noise); the
selection is Part I's: the support chosen by the bagged spread, and per
equation the threshold and the polynomial degree (1 to 5) chosen together by
BIC, with no true coefficients (the library does not contain them) and no
validation data. It writes the metric table and, beside
it, every run's choices with the last run's forecasts.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from mfsindy.experiments import cli_overrides
from mfsindy.cases.double_pendulum_forecasting import (
    DoublePendulumForecastConfig,
    run_double_pendulum_forecast_experiment,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Candidate supports, as divisors of the training horizon: the ODE ladder of
#: Part I, T/100 to T/5 as for every case.
SUPPORT_DIVISORS = (100.0, 50.0, 20.0, 10.0, 5.0)


def make_config(**overrides) -> DoublePendulumForecastConfig:
    settings = dict(n_runs=100, results_dir=str(RESULTS_DIR))
    settings.update(overrides)
    cfg = DoublePendulumForecastConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = [cfg.T_train / d for d in SUPPORT_DIVISORS]
    return cfg


def main() -> None:
    cfg = make_config(**cli_overrides())
    print(f"running {cfg.n_runs} Monte Carlo runs over {len(cfg.H_grid)} supports")
    metrics = run_double_pendulum_forecast_experiment(cfg)
    print(metrics.groupby(["model", "metric"])["value"].median().unstack().round(4))


if __name__ == "__main__":
    main()
