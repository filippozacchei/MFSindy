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

import argparse
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


def results_filename(lf_level: float, estimated: bool) -> str:
    """The metric table of one configuration. The main one -- LF level 0.25,
    true variances -- keeps the plain name the figures read."""

    if lf_level == DoublePendulumForecastConfig.noise_lf_level and not estimated:
        return DoublePendulumForecastConfig.results_filename
    return f"double_pendulum_part3_lf{lf_level:g}{'_estimated' if estimated else ''}_metrics.csv"


def make_config(**overrides) -> DoublePendulumForecastConfig:
    settings = dict(n_runs=100, results_dir=str(RESULTS_DIR))
    settings.update(overrides)
    settings.setdefault("results_filename", results_filename(
        settings.get("noise_lf_level", DoublePendulumForecastConfig.noise_lf_level),
        settings.get("estimate_noise", False)))
    cfg = DoublePendulumForecastConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = [cfg.T_train / d for d in SUPPORT_DIVISORS]
    return cfg


def main() -> None:
    # The LF noise level and whether the variances are estimated, for the noise
    # study; --runs and --jobs as for every case.
    parser = argparse.ArgumentParser()
    parser.add_argument("--lf-level", type=float, default=DoublePendulumForecastConfig.noise_lf_level)
    parser.add_argument("--estimated", action="store_true",
                        help="whiten with variances estimated from the data")
    args, rest = parser.parse_known_args()
    cfg = make_config(noise_lf_level=args.lf_level, estimate_noise=args.estimated, **cli_overrides(rest))
    print(f"running {cfg.n_runs} Monte Carlo runs over {len(cfg.H_grid)} supports: "
          f"LF level {cfg.noise_lf_level}, {'estimated' if cfg.estimate_noise else 'true'} variances "
          f"-> {cfg.results_filename}")
    metrics = run_double_pendulum_forecast_experiment(cfg)
    print(metrics.groupby(["model", "metric"])["value"].median().unstack().round(4))


if __name__ == "__main__":
    main()
