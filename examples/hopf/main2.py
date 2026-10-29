"""Part II pipeline for the Hopf benchmark, as a script.

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
from mfsindy.cases.hopf import (
    hopf_run_trajectories,
    _hopf_jacobian_norm,
    HopfIntraTrajectoryGLSConfig,
    _build_hopf_gls_artifacts,
    run_hopf_intra_trajectory_gls_experiment,
)

from mfsindy.weighted_weak_pde_library import weak_system, weak_validity_ratio

from mfsindy.experiments import (
    REFERENCE_SEED_OFFSET,
    build_polynomial_intra_trajectory_artifacts,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ERRORS_PATH = RESULTS_DIR / "hopf_part2_errors.csv"

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


def make_config(**overrides) -> HopfIntraTrajectoryGLSConfig:
    settings = dict(
        n_runs=100,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    cfg = HopfIntraTrajectoryGLSConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = supports(cfg)
    return cfg


def _design_artifacts(cfg, run_idx: int):
    """The first trajectory of run run_idx and its four libraries at cfg's
    support, seeded as the Monte Carlo seeds that run (every trajectory of a run
    shares the test functions)."""

    artifacts = _build_hopf_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx))
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
            library.fit([artifacts.data])
            weak_system(library, artifacts.data)
            row = {
                "K": int(library.K),
                "duplicates_dropped": int(getattr(library, "n_duplicate_domains_", 0)),
            }
            # One factor per equation when the noise is given per component.
            chols = getattr(library, "_L_chols", None) or [getattr(library, "_L_chol", None)]
            if chols[0] is not None:
                diagonals = [np.diag(L @ L.T) for L in chols]
                row.update(
                    diag_spread=float(max(d.max() / d.min() for d in diagonals)),
                    cov_cond=float(library.cov_cond_),
                    cov_rank_deficit=int(library.cov_rank_deficit_),
                )
            rows.append(dict(H=str(H), method=method, **row))
    return pd.DataFrame(rows).set_index(["H", "method"])


def supports(cfg) -> list:
    horizon = cfg.t1 - cfg.t0
    return [horizon / d for d in SUPPORT_DIVISORS]


#: Candidate supports, as divisors of the horizon.
#: Every rung picks its support from supports(cfg) by the spread of its own
#: ensemble, with the threshold set by BIC inside the ensemble
#: (mfsindy.experiments.selection): the same procedure for every rung, and no
#: true coefficients involved. The kappa table covers the same supports.
#: T/20 leaves 20 test functions per trajectory, at least twice the largest
#: library (9 terms). T/10 and T/5 go past it; a support is admissible for a
#: model only if its weak system has at least twice as many rows as library
#: terms (selection.MIN_ROWS_PER_TERM), which rules them out for one HF trajectory.
SUPPORT_DIVISORS = (200.0, 100.0, 50.0, 20.0, 10.0, 5.0)


def run_trajectories(cfg, run_idx: int = 0):
    """Run run_idx's trajectories as the Monte Carlo draws them:
    ``(t, clean, noisy, variance)``, the last three one entry per trajectory."""

    return hopf_run_trajectories(cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx))


def reference_trajectory(cfg):
    """One noisy trajectory for the kappa table, at the Monte Carlo's settings
    and a seed disjoint from its runs."""

    rng = np.random.default_rng(cfg.seed_base + REFERENCE_SEED_OFFSET)
    t_eval, _, noisy, variance = hopf_run_trajectories(cfg, rng)
    return noisy[0], t_eval, variance[0]


def _artifacts(cfg, data, t_grid, variance, H_xt, weak_seed):
    # K follows the support at coverage 2, the same rule the Monte Carlo builder
    # applies, here on whichever segment is passed. Left to the library it would
    # default to 100 regardless of H, and the kappa table would then describe
    # a design the run never uses.
    extent = float(np.max(t_grid) - np.min(t_grid))
    return build_polynomial_intra_trajectory_artifacts(
        data,
        t_grid,
        variance=variance,
        poly_degree=cfg.poly_degree,
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        K=max(2, int(round(extent / H_xt))),
        H_xt=H_xt,
        p=cfg.p,
        weak_seed=weak_seed,
    )


def kappa_by_support(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Kappa per candidate fitting support, on the reference trajectory.

    Part I averages this over several clean reference trajectories, because
    kappa depends on the trajectory through |grad F(u)| and a single draw makes
    it a function of the seed. There is one trajectory here, so what varies
    instead is sigma: the noise is heteroscedastic, so it does not cancel
    between the two terms of the ratio the way it does in Part I, and it is
    carried explicitly. ``kappa_min``/``kappa_max`` bracket the median over the
    test functions of one support, not over trajectories.
    """

    train, t_train, variance_train = reference_trajectory(cfg)
    jacobian = _hopf_jacobian_norm(train, cfg)
    # kappa takes one noise level per sample: the components' RMS.
    sigma = np.sqrt(np.mean(variance_train, axis=1))
    seed = cfg.seed_base + REFERENCE_SEED_OFFSET + 2

    rows = []
    for H in supports(cfg):
        library = _artifacts(
            cfg, train, t_train, variance_train, H, seed
        ).libraries["No weighting"]
        library.fit_transform([train])
        ratios = weak_validity_ratio(library, jacobian, sigma=sigma)
        rows.append(
            dict(
                H_xt=H,
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
    errors, _, _ = run_hopf_intra_trajectory_gls_experiment(cfg)
    if verbose:
        print(errors.groupby(["method", "metric"])["value"].median().unstack().round(5))
    return errors


def main() -> None:
    run_monte_carlo(make_config(**cli_overrides()))


if __name__ == "__main__":
    main()
