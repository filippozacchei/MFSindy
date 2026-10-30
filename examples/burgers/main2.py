"""Part II pipeline for the Burgers benchmark, as a script.

``python main2.py`` regenerates everything ``part2.ipynb`` displays: the weak
design each weighting fits, and the Monte Carlo error table. The notebook imports
these same functions, so there is one copy of the pipeline rather than two that
drift; it can either drive them or load what this script wrote.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from mfsindy.experiments import cli_overrides
from mfsindy.cases.burgers import (
    BurgersConfig,
    _make_burgers_gls_artifacts,
    build_true_burgers_coefficients,
    burgers_run_trajectories,
    run_burgers_experiment,
)

from mfsindy.weighted_weak_pde_library import (
    pde_sensitivity_fields,
    pde_weak_validity_ratio,
)

from mfsindy.experiments import (
    REFERENCE_SEED_OFFSET,
    build_pde_intra_trajectory_artifacts,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ERRORS_PATH = RESULTS_DIR / "burgers_part2_errors.csv"

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


def make_config(**overrides) -> BurgersConfig:
    settings = dict(
        n_runs=100,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    cfg = BurgersConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = supports(cfg)
    return cfg


def _design_artifacts(cfg, run_idx: int):
    """The first field of run run_idx and its four libraries at cfg's support,
    seeded as the Monte Carlo seeds that run (every field of a run shares the
    test functions)."""

    artifacts = _make_burgers_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx))
    return artifacts[0]


def weak_design(cfg, *, run_idx: int = 0) -> pd.DataFrame:
    """What each weighting actually fits, at every candidate support.

    All four rungs share one set of test functions, so a difference between them
    is the weighting and nothing else. The diagonal spread is what makes the
    diagonal rung worth having: it is 1 for the unit-variance covariance by
    construction, and well above 1 wherever the noise varies along the trajectory.
    """

    rows = []
    for H in cfg.H_grid:
        artifacts = _design_artifacts(replace(cfg, H_xt=H, K=None), run_idx)
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
            rows.append(dict(H=str(H), method=method, **row))
    return pd.DataFrame(rows).set_index(["H", "method"])


#: Candidate supports [h_x, h_t]. In time, the ODE divisors of the horizon
#: without T/200; in space, divisors of the domain 2L. The narrowest of each
#: are left out for cost: they set K, and the covariance grows as K^2. Every rung picks
#: among them by the spread of its own ensemble, with the threshold set by BIC
#: inside the ensemble (mfsindy.experiments.selection): the same procedure for
#: every rung, and no true coefficients involved. A support is admissible for a
#: rung only if its weak system has at least twice as many rows as library terms
#: (selection.MIN_ROWS_PER_TERM); with ten fields per run every support passes.
SPACE_DIVISORS = (16.0, 8.0, 4.0)
SUPPORT_DIVISORS = (100.0, 50.0, 20.0, 10.0, 5.0)


def supports(cfg) -> list:
    horizon = cfg.t1 - cfg.t0
    return [[2.0 * cfg.L / ds, horizon / d] for ds in SPACE_DIVISORS for d in SUPPORT_DIVISORS]


def run_trajectories(cfg, run_idx: int = 0):
    """Run run_idx's fields as the Monte Carlo draws them:
    ``(x, t, clean, noisy, variance)``, the last three one entry per field."""

    return burgers_run_trajectories(cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx))


def reference_trajectory(cfg):
    """One noisy field for the kappa table, at the Monte Carlo's settings
    and a seed disjoint from its runs."""

    rng = np.random.default_rng(cfg.seed_base + REFERENCE_SEED_OFFSET)
    x, t, _, noisy, variance = burgers_run_trajectories(cfg, rng)
    X, T = np.meshgrid(x, t)
    return noisy[0], np.asarray([X, T]).T, t, variance[0]


def reference_window(cfg) -> dict:
    """The reference field in the window layout the artifact builder takes."""

    noisy, grid, t_grid, variance = reference_trajectory(cfg)
    return {"data": noisy, "grid": grid, "t_grid": t_grid, "variance": variance}


def _artifacts(cfg, window, H_xt, weak_seed):
    """The four weightings on one window of the field, K following the support."""

    return build_pde_intra_trajectory_artifacts(
        window["data"],
        window["grid"],
        t_grid=window["t_grid"],
        variance=window["variance"],
        poly_degree=cfg.poly_degree,
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        function_include_bias=False,
        H_xt=H_xt,
        p=cfg.p,
        weak_seed=weak_seed,
        true_coefficients=build_true_burgers_coefficients(cfg.nu),
    )


def kappa_by_support(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Kappa per candidate fitting support, on the reference field.

    Dimensionless, so the numbers are comparable across supports and against the
    ODE cases; the scaling form h_t^2 h_x^-2m carries units and is not. Sigma is
    carried because the noise here varies over the field and so does not cancel
    between the two terms of the ratio.
    """

    train = reference_window(cfg)
    sigma = np.sqrt(train["variance"])
    coefficients = build_true_burgers_coefficients(cfg.nu)
    seed = cfg.seed_base + REFERENCE_SEED_OFFSET + 2

    rows = []
    for H in supports(cfg):
        library = _artifacts(cfg, train, H, seed).libraries["No weighting"]
        library.fit_transform([train["data"]])
        ratios = pde_weak_validity_ratio(
            library,
            pde_sensitivity_fields(library, train["data"], coefficients),
            sigma=sigma,
        )
        rows.append(
            dict(
                H_x=H[0],
                H_t=H[1],
                K=int(library.K),
                kappa_median=float(np.median(ratios)),
                kappa_min=float(np.min(ratios)),
                kappa_max=float(np.max(ratios)),
            )
        )
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
    return table


def run_monte_carlo(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Run the Monte Carlo, every rung on cfg's design.

    The runner writes the error table to ``cfg.results_dir``.
    """

    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs")
    errors, _, _ = run_burgers_experiment(cfg)
    if verbose:
        print(errors.groupby(["method", "metric"])["value"].median().unstack().round(5))
    return errors


def main() -> None:
    run_monte_carlo(make_config(**cli_overrides()))


if __name__ == "__main__":
    main()
