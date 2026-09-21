"""Part I pipeline for the hopf benchmark, as a script.

``python main1.py`` regenerates everything ``part1.ipynb`` displays: the tuning
record (selection, score surface, kappa table, weak design) and the Monte Carlo
error table. The notebook imports these same functions, so there is one copy of
the pipeline rather than two that drift; it can either drive them or load what
this script wrote.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from mfsindy.cases.hopf import (
    HopfMultiTrajectoryGLSConfig,
    build_hopf_weak_validation_blocks,
    fit_hopf_multi_trajectory_rollout_models,
    generate_hopf_dataset,
    hopf_kappa_by_support,
    hopf_validation_support,
    hopf_weak_design,
    run_hopf_multi_trajectory_gls_experiment,
)
from mfsindy.experiments import (
    FULL_COVARIANCE_RUNGS,
    TUNING_SEED_OFFSET,
    TuningArtifacts,
    evaluate_weak_form_models,
    load_tuning,
    run_with_tuned_configs,
    save_tuning,
    split_trajectory_list,
    tune_rungs,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TUNING_PATH = RESULTS_DIR / "hopf_part1_tuning.json"
ERRORS_PATH = RESULTS_DIR / "hopf_part1_errors.csv"

MODELS = ["HF", "LF", "MF", "VHF", "VLF", "PMF", "VMF", "MF_w"]

#: Takes per grid point. Each shares one dataset and differs only in where the
#: test functions land, so a selection is not decided by a single placement.
N_TAKES = 5


#: Floor on the noise ceiling of the validation target.
MIN_CEILING = 0.99


def make_config(**overrides) -> HopfMultiTrajectoryGLSConfig:
    settings = dict(
        p=2,
        n_runs=100,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    return HopfMultiTrajectoryGLSConfig(**settings)


def support_candidates(cfg) -> list[float]:
    return [cfg.T_train / d for d in (200.0, 100.0, 50.0, 20.0, 10.0, 5.0)]


def search_grid(cfg) -> dict:
    return {
        # Log-spaced and wide: with three points most rungs sat on an edge,
        # so the selection was truncated rather than chosen. With a grid
        # spanning 0.01-5 the best threshold is interior for every rung
        # tested. Every true coefficient is 1.0, so the grid stops below it.
        "stlsq_threshold": [0.02, 0.05, 0.1, 0.2, 0.5],
        "H_xt": support_candidates(cfg),
    }


def tune(cfg, *, results_dir: Path = RESULTS_DIR, verbose: bool = True) -> TuningArtifacts:
    """Select hyperparameters per rung and write the record to JSON."""

    results_dir = Path(results_dir)
    grid = search_grid(cfg)
    candidates = grid["H_xt"]

    reference, _, _ = generate_hopf_dataset(
        n_traj=1, T=cfg.T_true, dt=cfg.dt, noise_level=0.0,
        seed=cfg.seed_base,
        mu=cfg.mu,
        omega=cfg.omega,
    )
    tune_std = float(np.std(reference[0]))

    hf_tune, t_tune, _ = generate_hopf_dataset(
        n_traj=cfg.n_hf + 1,  # +1 held out to validate
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=cfg.noise_hf_rel * tune_std,
        seed=cfg.seed_base + TUNING_SEED_OFFSET,
        mu=cfg.mu,
        omega=cfg.omega,
    )
    lf_tune, _, _ = generate_hopf_dataset(
        n_traj=cfg.n_lf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=cfg.noise_lf_rel * tune_std,
        seed=cfg.seed_base + TUNING_SEED_OFFSET + 1,
        mu=cfg.mu,
        omega=cfg.omega,
    )
    hf_train, hf_val = split_trajectory_list(
        hf_tune, validation_fraction=1 / len(hf_tune)
    )
    take_seeds = [cfg.seed_base + TUNING_SEED_OFFSET + 2 + t for t in range(N_TAKES)]

    # One fixed validation system for the whole grid. Letting it follow the
    # candidate would give every point its own Theta, its own b and its own R^2
    # denominator, which is not a ranking. The noise ceiling bounds the choice:
    # b inherits observation noise through ||phi_dot||, so too narrow a support
    # caps the score any model can reach. Kappa is reported but does not bind --
    # this library is unweighted, so it invokes no covariance model.
    support = hopf_validation_support(
        cfg,
        hf_val[0],
        t_grid=t_tune,
        sigma=cfg.noise_hf_rel * tune_std,
        candidates=candidates,
        weak_seed=take_seeds[0] + 10,
        min_ceiling=MIN_CEILING,
    )
    if verbose:
        print(support.table.to_string(index=False))
        print("\nvalidation system:", support.summary())

    val_blocks = {
        seed: build_hopf_weak_validation_blocks(
            cfg, hf_val, t_grid=t_tune, H_val=support.H_val, weak_seed=seed + 10
        )
        for seed in take_seeds
    }

    # Kappa is reported, not enforced. It is defined at the true coefficients --
    # (A.1) expands the residual at Xi* -- so gating the search on it would leak
    # information no practitioner has, and would make this selection protocol
    # unreproducible on real data. Everything else here is oracle-free: the
    # metric scores held-out noisy data and the validation support follows a
    # noise ceiling computable from sigma. So kappa stays a diagnostic, and a
    # rung landing where the covariance model is invalid is a result to report
    # rather than a search to constrain.
    kappa_table = hopf_kappa_by_support(cfg, candidates)
    if verbose:
        print(kappa_table.to_string(index=False))

    selections, score_table = tune_rungs(
        cfg,
        param_grid=grid,
        rungs=MODELS,
        metric="weak_r2",
        reducer=lambda scores: float(np.mean(np.clip(scores, -1.0, 1.0))),
        evaluate=lambda candidate: pd.concat(
            [
                evaluate_weak_form_models(
                    {
                        rung: model.coefficients
                        for rung, model in fit_hopf_multi_trajectory_rollout_models(
                            candidate,
                            hf_trajectories=hf_train,
                            lf_trajectories=lf_tune,
                            t_grid=t_tune,
                            weak_seed=seed,
                        ).items()
                    },
                    val_blocks[seed],
                )
                for seed in take_seeds
            ],
            ignore_index=True,
        ),
    )

    # Where each rung actually landed, so the tables can show whether the
    # covariance model held at the support the search chose.
    kappa_by_support = {
        tuple(np.atleast_1d(row.H_xt)): float(row.kappa_median)
        for row in kappa_table.itertuples()
    }
    kappa_at_selection = {
        rung: kappa_by_support[tuple(np.atleast_1d(sel.best_params["H_xt"]))]
        for rung, sel in selections.items()
    }
    if verbose:
        print(f"\nkappa at each selection: "
              + ", ".join(f"{r}={k:.3g}" for r, k in kappa_at_selection.items()))

    weak_design = {}
    for rung, selection in selections.items():
        rung_cfg = deepcopy(cfg)
        for name, value in selection.best_params.items():
            setattr(rung_cfg, name, value)
        weak_design[rung] = hopf_weak_design(rung_cfg)

    save_tuning(
        results_dir / TUNING_PATH.name,
        selections=selections,
        param_grid=grid,
        table=score_table,
        kappa_table=kappa_table,
        fixed={
            "weak_design": weak_design,
            "n_takes": N_TAKES,
            # K is derived from the support at coverage 2; the realised numbers
            # are in weak_design. Recording a rule as well only let the two
            # drift apart, which they had.
            "K_rule": "max(2, round((t1 - t0) / H_xt))  # coverage 2",
            "validation_metric": "weak_r2",
            "kappa_diagnostic": {
                "enforced": False,
                "statistic": "kappa_median (median over test functions),"
                             " median over fixed reference trajectories",
                "relevant_to": sorted(FULL_COVARIANCE_RUNGS),
                "note": "defined at the true coefficients, so reported"
                        " a posteriori rather than used to select",
                "kappa_at_selection": kappa_at_selection,
            },
            "validation_support": {
                "H_val": support.H_val,
                "K": support.K,
                "ceiling": support.ceiling,
                "kappa": support.kappa,  # reported, not enforced
                "min_ceiling": MIN_CEILING,
            },
            "poly_degree": cfg.poly_degree,
            "mu": cfg.mu,
            "omega": cfg.omega,
            "p": cfg.p,
            "n_ensemble_models": cfg.n_ensemble_models,
            "T_train": cfg.T_train,
            "T_true": cfg.T_true,
            "dt": cfg.dt,
            "n_runs": cfg.n_runs,
            "n_hf": cfg.n_hf,
            "n_lf": cfg.n_lf,
            "noise_hf_rel": cfg.noise_hf_rel,
            "noise_lf_rel": cfg.noise_lf_rel,
            "deduplicate": cfg.deduplicate,
            "seed_base": cfg.seed_base,
        },
    )
    return load_tuning(results_dir / TUNING_PATH.name)


def run_monte_carlo(cfg, selections, *, results_dir: Path = RESULTS_DIR,
                    verbose: bool = True) -> pd.DataFrame:
    """Run the Monte Carlo once per rung, each under its own tuned config."""

    results_dir = Path(results_dir)
    run_cfgs = {}
    for rung, selection in selections.items():
        per_rung = deepcopy(cfg)
        for name, value in selection.best_params.items():
            setattr(per_rung, name, value)
        per_rung.results_filename = f"hopf_part1_{rung.lower()}_errors.csv"
        run_cfgs[rung] = per_rung

    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs per rung")
    errors = run_with_tuned_configs(run_hopf_multi_trajectory_gls_experiment, run_cfgs)
    errors.to_csv(results_dir / ERRORS_PATH.name, index=False)
    if verbose:
        print(errors.groupby(["model", "metric"])["value"].median().unstack().round(4))
    return errors


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_config()
    artifacts = tune(cfg)
    run_monte_carlo(cfg, artifacts.selections)


if __name__ == "__main__":
    main()
