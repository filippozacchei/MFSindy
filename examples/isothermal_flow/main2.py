"""Part II pipeline for the Isothermal flow benchmark, as a script.

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
from mfsindy.cases.isothermal_flow import (
    NSIsothermalIntraTrajectoryGLSConfig,
    _make_ns_gls_artifacts,
    build_ns_artifacts,
    build_true_ns_isothermal_coefficients,
    generate_isothermal_ns_dataset,
    get_ns_isothermal_feature_names,
    ns_monomial_fields,
    ns_run_trajectories,
    ns_target_variance,
    run_ns_isothermal_intra_trajectory_gls_experiment,
)

from mfsindy.weighted_weak_pde_library import (
    pde_sensitivity_fields,
    pde_weak_validity_ratio,
    weak_system,
)

from mfsindy.experiments import REFERENCE_SEED_OFFSET

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ERRORS_PATH = RESULTS_DIR / "navierstokes_part2_errors.csv"

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


def make_config(**overrides) -> NSIsothermalIntraTrajectoryGLSConfig:
    settings = dict(
        n_runs=25,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    cfg = NSIsothermalIntraTrajectoryGLSConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = supports(cfg)
    return cfg


def _truth(cfg):
    """The true coefficients the runner builds.

    Reproduced here rather than imported because the runner assembles them
    inside itself; weak_design and kappa_by_support need them, and they are
    fixed by cfg alone, so both get the identical reference.
    """

    U_clean_ref, _, grid_ref = generate_isothermal_ns_dataset(
        N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
        seed=cfg.seed_base,
    )
    coefficients = build_true_ns_isothermal_coefficients(
        get_ns_isothermal_feature_names(
            cfg, grid=grid_ref, reference_trajectory=U_clean_ref
        ),
        RT=cfg.RT,
        mu=cfg.mu,
    )
    return coefficients


def _design_artifacts(cfg, run_idx: int):
    """The first flow of run run_idx and its four libraries at cfg's support,
    seeded as the Monte Carlo seeds that run (every flow of a run shares the
    test functions)."""

    coefficients = _truth(cfg)
    artifacts = _make_ns_gls_artifacts(
        run_idx,
        cfg,
        np.random.default_rng(cfg.seed_base + 100 * run_idx),
        true_coefficients=coefficients,
    )
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


#: Candidate supports, as divisors of the side L (space, both axes) and the
#: horizon T (time), the same as Part I. Every rung picks its support from
#: supports(cfg) by the spread of its own ensemble, with the threshold set by BIC
#: inside the ensemble (mfsindy.experiments.selection): the same procedure for
#: every rung, and no true coefficients involved. The kappa table covers the
#: same supports.
SPACE_DIVISORS = (10.0, 5.0)
TIME_DIVISORS = (20.0, 10.0, 5.0)


def supports(cfg) -> list:
    return [[cfg.L / ds, cfg.L / ds, cfg.T / dt] for ds in SPACE_DIVISORS for dt in TIME_DIVISORS]


def run_trajectories(cfg, run_idx: int = 0):
    """Run run_idx's flows as the Monte Carlo draws them:
    ``(t, grid, clean, noisy, variance)``, the last three one entry per flow."""

    return ns_run_trajectories(cfg, np.random.default_rng(cfg.seed_base + 100 * run_idx))


def reference_window(cfg) -> dict:
    """One noisy flow for the kappa table, at the Monte Carlo's settings and a
    seed disjoint from its runs, with its targets' noise variance."""

    t, grid, clean, noisy, variance = ns_run_trajectories(
        cfg, np.random.default_rng(cfg.seed_base + REFERENCE_SEED_OFFSET)
    )
    return {
        "data": noisy[0],
        "grid": grid,
        "t_grid": t,
        "variance": ns_target_variance(clean[0], variance[0]),
    }


def kappa_by_support(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Kappa per candidate fitting support, on the reference flow.

    Dimensionless and sigma-weighted, as for Burgers.
    """

    coefficients = _truth(cfg)
    train = reference_window(cfg)
    # kappa takes one noise level per point: the components' RMS.
    sigma = np.sqrt(np.mean(train["variance"], axis=-1))
    seed = cfg.seed_base + REFERENCE_SEED_OFFSET + 2

    rows = []
    for H in supports(cfg):
        library = build_ns_artifacts(
            train["data"], train["grid"], train["t_grid"], train["variance"], cfg,
            weak_seed=seed, true_coefficients=coefficients, H_xt=H,
        ).libraries["No weighting"]
        library.fit([train["data"]])
        # The library's inputs are the monomial fields, so the sensitivity is
        # taken with respect to them (see ns_isothermal_kappa_by_support).
        ratios = pde_weak_validity_ratio(
            library,
            pde_sensitivity_fields(library, ns_monomial_fields(train["data"]), coefficients),
            sigma=sigma,
        )
        rows.append(
            dict(
                H_s=H[0],
                H_t=H[2],
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
    errors, _, _ = run_ns_isothermal_intra_trajectory_gls_experiment(cfg)
    if verbose:
        print(errors.groupby(["method", "metric"])["value"].median().unstack().round(5))
    return errors


def main() -> None:
    run_monte_carlo(make_config(**cli_overrides()))


if __name__ == "__main__":
    main()
