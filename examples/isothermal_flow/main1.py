"""Part I pipeline for the Isothermal flow benchmark, as a script.

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

import numpy as np
import pandas as pd

from mfsindy.experiments import cli_overrides
from mfsindy.cases.isothermal_flow import (
    NSIsothermalMultiTrajectoryGLSConfig,
    build_true_ns_isothermal_coefficients,
    generate_isothermal_ns_dataset,
    get_ns_isothermal_feature_names,
    ns_isothermal_kappa_by_support,
    ns_isothermal_weak_design,
    run_ns_isothermal_multi_trajectory_gls_experiment,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ERRORS_PATH = RESULTS_DIR / "navierstokes_part1_errors.csv"

MODELS = ["HF", "LF", "MF", "VHF", "VLF", "PMF", "VMF", "MF_w"]

#: Candidate supports, as divisors of the training horizon and domain. Every model picks
#: among them by the spread of its own ensemble, with the threshold set by BIC
#: inside the ensemble (mfsindy.experiments.selection): the same procedure for
#: every model, and no true coefficients involved. In space the grid starts at
#: L/10 (L/20 would be about three grid spacings at N = 64). A support is
#: admissible for a model only if its weak system has at least twice as many rows
#: as library terms (selection.MIN_ROWS_PER_TERM), 228 for the 114-term library.
SPACE_DIVISORS = (10.0, 5.0)
TIME_DIVISORS = (20.0, 10.0, 5.0)


def make_config(**overrides) -> NSIsothermalMultiTrajectoryGLSConfig:
    settings = dict(
        n_runs=25,
        n_lf=10,
        n_hf=1,
        p=2,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    cfg = NSIsothermalMultiTrajectoryGLSConfig(**settings)
    if cfg.H_grid is None:
        cfg.H_grid = [[cfg.L / dx, cfg.L / dx, cfg.T / dt] for dx in SPACE_DIVISORS for dt in TIME_DIVISORS]
    return cfg


def design_report(cfg) -> pd.DataFrame:
    """The weak design at every candidate support, with kappa there.

    Kappa is the median over clean reference trajectories, as in
    :func:`ns_isothermal_kappa_by_support`; it is defined at the true coefficients,
    so it is reported, not used to choose anything.
    """

    # Kappa is computed at the true coefficients, which are indexed by the
    # library's feature names; those need one solved flow to be read off.
    reference, _, grid = generate_isothermal_ns_dataset(
        N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
        seed=cfg.seed_base,
    )
    coefficients = build_true_ns_isothermal_coefficients(
        get_ns_isothermal_feature_names(
            cfg, grid=np.asarray(grid, dtype=float), reference_trajectory=reference
        ),
        RT=cfg.RT,
        mu=cfg.mu,
    )
    kappa = ns_isothermal_kappa_by_support(
        cfg, cfg.H_grid, coefficients=coefficients, n_reference=3
    )
    design = pd.DataFrame([ns_isothermal_weak_design(replace(cfg, H_xt=H, K=None)) for H in cfg.H_grid])
    return pd.concat([kappa.reset_index(drop=True), design.drop(columns="K", errors="ignore")], axis=1)


def run_monte_carlo(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Run the Monte Carlo; every model selects its own support and threshold.

    The runner writes the error table to ``cfg.results_dir``, and beside it the
    last run's selection (``mfsindy.experiments.load_selection``).
    """

    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs over {len(cfg.H_grid)} supports")
    errors = run_ns_isothermal_multi_trajectory_gls_experiment(cfg)[0]
    if verbose:
        print(errors.groupby(["model", "metric"])["value"].median().unstack().round(4))
    return errors


def main() -> None:
    cfg = make_config(**cli_overrides())
    print(design_report(cfg).to_string(index=False))
    run_monte_carlo(cfg)


if __name__ == "__main__":
    main()
