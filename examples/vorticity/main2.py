"""Part II pipeline for the vorticity-flow benchmark, as a script.

``python main2.py`` runs the Monte Carlo of the four weightings and writes the
error table, as for the isothermal flow.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mfsindy.experiments import REFERENCE_SEED_OFFSET, cli_overrides
from mfsindy.cases.vorticity_flow import (
    VorticityIntraTrajectoryGLSConfig,
    build_true_vorticity_coefficients,
    build_vorticity_artifacts,
    generate_vorticity_dataset,
    get_vorticity_feature_names,
    run_vorticity_intra_trajectory_gls_experiment,
    vorticity_monomial_fields,
    vorticity_run_trajectories,
)
from mfsindy.weighted_weak_pde_library import pde_sensitivity_fields, pde_weak_validity_ratio

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


def kappa_by_support(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Kappa per candidate support, on one window of a reference flow at the
    Monte Carlo's settings and a seed disjoint from its runs; sigma-weighted, as
    for the isothermal flow, with the vorticity's noise (the identified field)."""

    U_ref, _, grid_ref = generate_vorticity_dataset(N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, nu=cfg.nu,
                                                    ic=cfg.ic, seed=cfg.seed_base)
    coefficients = build_true_vorticity_coefficients(
        get_vorticity_feature_names(cfg, grid=grid_ref, reference_trajectory=U_ref), nu=cfg.nu)
    t, grid, _, noisy, variances = vorticity_run_trajectories(
        cfg, np.random.default_rng(cfg.seed_base + REFERENCE_SEED_OFFSET))
    data, variance = noisy[0], variances[0][..., 2:3]
    seed = cfg.seed_base + REFERENCE_SEED_OFFSET + 2

    rows = []
    for H in supports(cfg):
        library = build_vorticity_artifacts(data, grid, t, variance, cfg, weak_seed=seed,
                                            true_coefficients=coefficients, H_xt=H).libraries["No weighting"]
        library.fit([data])
        ratios = pde_weak_validity_ratio(
            library, pde_sensitivity_fields(library, vorticity_monomial_fields(data), coefficients),
            sigma=np.sqrt(variance[..., 0]))
        rows.append(dict(H_s=H[0], H_t=H[2], K=int(library.K), kappa_median=float(np.median(ratios)),
                         kappa_min=float(np.min(ratios)), kappa_max=float(np.max(ratios))))
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
    return table


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
