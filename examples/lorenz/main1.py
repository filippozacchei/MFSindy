"""Part I pipeline for the Lorenz benchmark, as a script.

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

from mfsindy.cases.lorenz import (
    LorenzMultiTrajectoryGLSConfig,
    build_lorenz_weak_validation_blocks,
    fit_lorenz_multi_trajectory_rollout_models,
    generate_lorenz_dataset,
    lorenz_kappa_by_support,
    lorenz_validation_support,
    lorenz_weak_design,
    run_lorenz_multi_trajectory_gls_experiment,
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
TUNING_PATH = RESULTS_DIR / "lorenz_part1_tuning.json"
ERRORS_PATH = RESULTS_DIR / "lorenz_part1_errors.csv"

MODELS = ["HF", "LF", "MF", "VHF", "VLF", "PMF", "VMF", "MF_w"]

#: Takes per grid point. Each shares one dataset and differs only in where the
#: test functions land, so a selection is not decided by a single placement.
N_TAKES = 5

#: Kappa bound for the rungs whitened by the full weak covariance. A guideline
#: on an approximation, not a theorem: a support sitting near it is marginal
#: rather than disqualified, which is why the table reports the range too.
KAPPA_MAX = 0.25

#: Floor on the noise ceiling of the validation target.
MIN_CEILING = 0.99


def make_config(**overrides) -> LorenzMultiTrajectoryGLSConfig:
    settings = dict(
        p=2,
        n_runs=100,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    return LorenzMultiTrajectoryGLSConfig(**settings)


def support_candidates(cfg) -> list[float]:
    return [cfg.T_train / d for d in (200.0, 100.0, 50.0, 20.0, 10.0, 5.0)]


def search_grid(cfg) -> dict:
    return {
        "stlsq_threshold": [0.1, 0.2, 0.5],
        "H_xt": support_candidates(cfg),
    }


def tune(cfg, *, results_dir: Path = RESULTS_DIR, verbose: bool = True) -> TuningArtifacts:
    """Select hyperparameters per rung and write the record to JSON."""

    results_dir = Path(results_dir)
    grid = search_grid(cfg)
    candidates = grid["H_xt"]

    reference, _, _ = generate_lorenz_dataset(
        n_traj=1, T=cfg.T_true, dt=cfg.dt, noise_level=0.0, seed=cfg.seed_base
    )
    tune_std = float(np.std(reference[0]))

    hf_tune, t_tune, _ = generate_lorenz_dataset(
        n_traj=cfg.n_hf + 1,  # +1 held out to validate
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=cfg.noise_hf_rel * tune_std,
        seed=cfg.seed_base + TUNING_SEED_OFFSET,
    )
    lf_tune, _, _ = generate_lorenz_dataset(
        n_traj=cfg.n_lf,
        T=cfg.T_train,
        dt=cfg.dt,
        noise_level=cfg.noise_lf_rel * tune_std,
        seed=cfg.seed_base + TUNING_SEED_OFFSET + 1,
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
    support = lorenz_validation_support(
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
        seed: build_lorenz_weak_validation_blocks(
            cfg, hf_val, t_grid=t_tune, H_val=support.H_val, weak_seed=seed + 10
        )
        for seed in take_seeds
    }

    # Kappa restricts only the rungs whitened by the full covariance. HF, LF and
    # MF form none; VMF uses the marginal variances alone, where a global
    # rescale cancels. Restricting every rung alike would shrink the baselines'
    # search space for a condition they never invoke.
    kappa_table = lorenz_kappa_by_support(cfg, candidates)
    kappa_table["admissible"] = kappa_table["kappa_median"] <= KAPPA_MAX
    if verbose:
        print(kappa_table.to_string(index=False))
    admissible_h = set(kappa_table.loc[kappa_table["admissible"], "H_xt"])

    def admissible(rung, params):
        if rung not in FULL_COVARIANCE_RUNGS:
            return True
        return params["H_xt"] in admissible_h

    selections, score_table = tune_rungs(
        cfg,
        param_grid=grid,
        rungs=MODELS,
        metric="weak_r2",
        reducer=lambda scores: float(np.mean(np.clip(scores, -1.0, 1.0))),
        admissible=admissible,
        evaluate=lambda candidate: pd.concat(
            [
                evaluate_weak_form_models(
                    {
                        rung: model.coefficients
                        for rung, model in fit_lorenz_multi_trajectory_rollout_models(
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

    weak_design = {}
    for rung, selection in selections.items():
        rung_cfg = deepcopy(cfg)
        for name, value in selection.best_params.items():
            setattr(rung_cfg, name, value)
        weak_design[rung] = lorenz_weak_design(rung_cfg)

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
            "kappa_gate": {
                "max_kappa": KAPPA_MAX,
                "statistic": "kappa_median (median over test functions),"
                             " worst case over fixed reference trajectories",
                "applies_to": sorted(FULL_COVARIANCE_RUNGS),
                "admissible_H_xt": sorted(admissible_h),
            },
            "validation_support": {
                "H_val": support.H_val,
                "K": support.K,
                "ceiling": support.ceiling,
                "kappa": support.kappa,  # reported, not enforced
                "min_ceiling": MIN_CEILING,
            },
            "poly_degree": cfg.poly_degree,
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
        per_rung.results_filename = f"lorenz_part1_{rung.lower()}_errors.csv"
        run_cfgs[rung] = per_rung

    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs per rung")
    errors = run_with_tuned_configs(run_lorenz_multi_trajectory_gls_experiment, run_cfgs)
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
