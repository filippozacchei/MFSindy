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
    _build_custom_library,
    _build_ns_gls_artifacts,
    add_heteroscedastic_noise_temporal_derivative,
    build_true_ns_isothermal_coefficients,
    generate_isothermal_ns_dataset,
    get_ns_isothermal_feature_names,
    run_ns_isothermal_intra_trajectory_gls_experiment,
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
    """The grid, feature library and true coefficients the runner builds.

    Reproduced here rather than imported because the runner assembles them
    inside itself; weak_design and kappa_by_support need the same three, and they are fixed by
    cfg alone, so both get the identical reference.
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
    return grid_ref, _build_custom_library(), coefficients


def _design_artifacts(cfg, run_idx: int):
    """Run run_idx's data and the four libraries at cfg's support, seeded as
    the Monte Carlo seeds that run."""

    # The builder needs the grid, the feature library and the true coefficients,
    # which the runner assembles inside itself; _truth rebuilds the same three
    # from cfg alone. Without them this function could not be called at all.
    grid, base_library, coefficients = _truth(cfg)
    artifacts = _build_ns_gls_artifacts(
        run_idx,
        cfg,
        np.random.default_rng(cfg.seed_base + 100 * run_idx),
        grid=grid,
        base_library=base_library,
        true_coefficients=coefficients,
    )
    return artifacts


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


#: Every rung picks its support from supports(cfg) by the spread of its own
#: ensemble, with the threshold set by BIC inside the ensemble
#: (mfsindy.experiments.selection): the same procedure for every rung, and no
#: true coefficients involved. The kappa table covers the same supports.
def supports(cfg) -> list:
    h_s = [cfg.L / d for d in (20.0, 10.0, 5.0)]
    h_t = [cfg.T / d for d in (80.0, 40.0, 20.0)]
    return [[hs, hs, ht] for hs in h_s for ht in h_t]


def reference_trajectory(cfg):
    """One noisy flow for the kappa table, at the Monte Carlo's settings
    and a seed disjoint from its runs."""

    U_clean, t, grid = generate_isothermal_ns_dataset(
        N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
        seed=cfg.seed_base + REFERENCE_SEED_OFFSET,
    )
    noisy, variance = add_heteroscedastic_noise_temporal_derivative(
        U_clean, t, sigma0=cfg.sigma0, alpha=cfg.alpha,
        rng=np.random.default_rng(cfg.seed_base + REFERENCE_SEED_OFFSET),
    )
    return noisy, np.asarray(grid, dtype=float), t, variance


def reference_window(cfg) -> dict:
    """The reference flow in the window layout the artifact builder takes."""

    noisy, grid, t_grid, variance = reference_trajectory(cfg)
    return {"data": noisy, "grid": grid, "t_grid": t_grid, "variance": variance}


def _artifacts(cfg, window, H_xt, weak_seed, *, base_library, coefficients):
    """The four weightings on one window of the flow, K following the support."""

    return build_pde_intra_trajectory_artifacts(
        window["data"],
        window["grid"],
        t_grid=window["t_grid"],
        variance=window["variance"] / np.mean(window["variance"]),
        function_library=base_library,
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        H_xt=H_xt,
        p=cfg.p,
        weak_seed=weak_seed,
        true_coefficients=coefficients,
    )


def kappa_by_support(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Kappa per candidate fitting support, on the reference flow.

    Dimensionless and sigma-weighted, as for Burgers.
    """

    _, base_library, coefficients = _truth(cfg)
    train = reference_window(cfg)
    sigma = np.sqrt(train["variance"])
    seed = cfg.seed_base + REFERENCE_SEED_OFFSET + 2

    rows = []
    for H in supports(cfg):
        library = _artifacts(
            cfg, train, H, seed, base_library=base_library, coefficients=coefficients
        ).libraries["No weighting"]
        library.fit_transform([train["data"]])
        ratios = pde_weak_validity_ratio(
            library,
            pde_sensitivity_fields(library, train["data"], coefficients),
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
