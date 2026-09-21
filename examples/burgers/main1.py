"""Part I pipeline for the Burgers benchmark, as a script.

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

from mfsindy.cases.burgers import (
    BurgersMultiTrajectoryGLSConfig,
    build_burgers_weak_validation_blocks,
    burgers_kappa_by_support,
    burgers_validation_support,
    burgers_weak_design,
    fit_burgers_multi_trajectory_coefficients,
    generate_burgers_dataset,
    run_burgers_multi_trajectory_gls_experiment,
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
TUNING_PATH = RESULTS_DIR / "burgers_part1_tuning.json"
ERRORS_PATH = RESULTS_DIR / "burgers_part1_errors.csv"

MODELS = ["HF", "LF", "MF", "VHF", "VLF", "PMF", "VMF", "MF_w"]
N_TAKES = 5
MIN_CEILING = 0.99


def make_config(**overrides) -> BurgersMultiTrajectoryGLSConfig:
    settings = dict(
        n_runs=100,
        n_lf=10,
        n_hf=1,
        p=2,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    return BurgersMultiTrajectoryGLSConfig(**settings)


def search_grid(cfg) -> dict:
    h_x = [cfg.L / d for d in (20.0, 10.0, 5.0)]
    h_t = [cfg.T_train / d for d in (20.0, 10.0, 5.0)]
    return {
        # Log-spaced and wide: with three points most rungs sat on an edge,
        # so the selection was truncated rather than chosen. With a grid
        # spanning 0.01-5 the best threshold is interior for every rung
        # tested. Extended down: every rung sat on the old lower edge of 0.01.
        "stlsq_threshold": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
        "H_xt": [[hx, ht] for hx in h_x for ht in h_t],
    }


def tune(cfg, *, results_dir: Path = RESULTS_DIR, verbose: bool = True) -> TuningArtifacts:
    """Select hyperparameters per rung and write the record to JSON."""

    results_dir = Path(results_dir)
    grid = search_grid(cfg)

    # The search DATA is drawn at TUNING_SEED_OFFSET, disjoint from the
    # evaluation seeds. The state_std that sets the noise level must NOT be:
    # the Monte Carlo scales its noise by the reference at cfg.seed_base, and
    # tuning at one SNR while evaluating at another is its own bug.
    _, t_search, x_search, _ = generate_burgers_dataset(
        n_traj=1, T=cfg.T_train, dt=cfg.dt, noise_level=0.0,
        seed=cfg.seed_base + TUNING_SEED_OFFSET, L=cfg.L, NX=cfg.NX, nu=cfg.nu,
    )
    reference, _, _, _ = generate_burgers_dataset(
        n_traj=1, T=cfg.T_train, dt=cfg.dt, noise_level=0.0,
        seed=cfg.seed_base, L=cfg.L, NX=cfg.NX, nu=cfg.nu,
    )
    state_std = float(np.std(reference[0]))
    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std

    hf_search, _, _, _ = generate_burgers_dataset(
        n_traj=cfg.n_hf + 1,  # +1 held out to validate
        T=cfg.T_train, dt=cfg.dt, noise_level=noise_hf_abs,
        seed=cfg.seed_base + TUNING_SEED_OFFSET, L=cfg.L, NX=cfg.NX, nu=cfg.nu,
    )
    lf_search, _, _, _ = generate_burgers_dataset(
        n_traj=cfg.n_lf, T=cfg.T_train, dt=cfg.dt, noise_level=noise_lf_abs,
        seed=cfg.seed_base + TUNING_SEED_OFFSET + 1, L=cfg.L, NX=cfg.NX, nu=cfg.nu,
    )
    hf_train, hf_val = split_trajectory_list(
        hf_search, validation_fraction=1 / len(hf_search)
    )
    take_seeds = [cfg.seed_base + TUNING_SEED_OFFSET + 2 + t for t in range(N_TAKES)]

    # One fixed validation system for the whole grid; letting it follow the
    # candidate would give every point its own target, which is not a ranking.
    support = burgers_validation_support(
        cfg, hf_val[0], t_grid=t_search, x_grid=x_search, sigma=noise_hf_abs,
        candidates=grid["H_xt"], weak_seed=take_seeds[0] + 10,
        min_ceiling=MIN_CEILING,
    )
    if verbose:
        print(support.table.to_string(index=False))
        print("\nvalidation system:", support.summary())

    val_blocks = {
        seed: build_burgers_weak_validation_blocks(
            cfg, hf_val, t_grid=t_search, x_grid=x_search,
            H_val=support.H_val, weak_seed=seed + 10, group_name="validation",
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
    kappa_table = burgers_kappa_by_support(cfg, grid["H_xt"])
    if verbose:
        print(kappa_table.to_string(index=False))

    def score_take(candidate_cfg, weak_seed):
        coef_map = fit_burgers_multi_trajectory_coefficients(
            candidate_cfg,
            hf_trajectories=hf_train,
            lf_trajectories=lf_search,
            t_grid=t_search,
            x_grid=x_search,
            noise_hf_abs=noise_hf_abs,
            noise_lf_abs=noise_lf_abs,
            weak_seed=weak_seed,
        )
        return evaluate_weak_form_models(coef_map, val_blocks[weak_seed])

    selections, score_table = tune_rungs(
        cfg,
        param_grid=grid,
        rungs=MODELS,
        metric="weak_r2",
        reducer=lambda scores: float(np.mean(np.clip(scores, -1.0, 1.0))),
        evaluate=lambda candidate: pd.concat(
            [score_take(candidate, seed) for seed in take_seeds], ignore_index=True
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
        weak_design[rung] = burgers_weak_design(rung_cfg)

    save_tuning(
        results_dir / TUNING_PATH.name,
        selections=selections,
        param_grid=grid,
        table=score_table,
        kappa_table=kappa_table,
        fixed={
            "weak_design": weak_design,
            "n_takes": N_TAKES,
            "validation_metric": "weak_r2 (per state component)",
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
            "K": cfg.K,
            "poly_degree": cfg.poly_degree,
            "derivative_order": cfg.derivative_order,
            "p": cfg.p,
            "n_ensemble_models": cfg.n_ensemble_models,
            "T_train": cfg.T_train,
            "dt": cfg.dt,
            "NX": cfg.NX,
            "L": cfg.L,
            "nu": cfg.nu,
            "n_runs": cfg.n_runs,
            "n_hf": cfg.n_hf,
            "n_lf": cfg.n_lf,
            "noise_hf_rel": cfg.noise_hf_rel,
            "noise_lf_rel": cfg.noise_lf_rel,
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
        per_rung.results_filename = f"burgers_part1_{rung.lower()}_errors.csv"
        run_cfgs[rung] = per_rung

    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs per rung")
    errors = run_with_tuned_configs(run_burgers_multi_trajectory_gls_experiment, run_cfgs)
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
