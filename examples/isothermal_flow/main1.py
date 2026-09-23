"""Part I pipeline for the isothermal-flow benchmark, as a script.

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

from mfsindy.cases.isothermal_flow import (
    NSIsothermalMultiTrajectoryGLSConfig,
    build_ns_isothermal_weak_validation_blocks,
    build_true_ns_isothermal_coefficients,
    fit_ns_isothermal_multi_trajectory_coefficients,
    generate_isothermal_ns_dataset,
    ns_isothermal_kappa_by_support,
    get_ns_isothermal_feature_names,
    ns_isothermal_validation_support,
    ns_isothermal_weak_design,
    run_ns_isothermal_multi_trajectory_gls_experiment,
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
TUNING_PATH = RESULTS_DIR / "navierstokes_part1_tuning.json"
ERRORS_PATH = RESULTS_DIR / "navierstokes_part1_errors.csv"

MODELS = ["HF", "LF", "MF", "VHF", "VLF", "PMF", "VMF", "MF_w"]
N_TAKES = 5
MIN_CEILING = 0.99

#: Reference flows for the kappa table. This case is the expensive one -- a
#: single 64x64 field at dt 1e-3 is 98 MB and takes over a second to solve --
#: so the spread is sampled with fewer draws than the ODE benchmarks use.
N_KAPPA_REFERENCE = 3


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
    return NSIsothermalMultiTrajectoryGLSConfig(**settings)


def search_grid(cfg) -> dict:
    h_xy = [cfg.L / 20.0, cfg.L / 10.0, cfg.L / 5.0]
    h_t = [cfg.T / 20.0, cfg.T / 10.0, cfg.T / 5.0]
    return {
        "stlsq_threshold": [0.1, 0.2, 0.5],
        "H_xt": [[h, h, ht] for h in h_xy for ht in h_t],
    }


def tune(cfg, *, results_dir: Path = RESULTS_DIR, verbose: bool = True) -> TuningArtifacts:
    """Select hyperparameters per rung and write the record to JSON."""

    results_dir = Path(results_dir)
    grid = search_grid(cfg)

    # The search DATA is drawn at TUNING_SEED_OFFSET, disjoint from the
    # evaluation seeds. The state_std that sets the noise level must NOT be:
    # the Monte Carlo scales its noise by the reference at cfg.seed_base, and
    # make_initial_condition randomises the Taylor-Green amplitudes AND
    # wavenumbers, so a reference drawn at the offset is a different flow.
    _, t_search, grid_search = generate_isothermal_ns_dataset(
        N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
        seed=cfg.seed_base + TUNING_SEED_OFFSET, 
    )
    reference, _, _ = generate_isothermal_ns_dataset(
        N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
        seed=cfg.seed_base,
    )
    state_std = float(np.std(reference[:, :, :, 2]))
    noise_hf_abs = cfg.noise_hf_rel * state_std
    noise_lf_abs = cfg.noise_lf_rel * state_std

    def sample(seed: int, noise_abs: float, *, noise_seed: int) -> np.ndarray:
        clean, _, _ = generate_isothermal_ns_dataset(
            N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
            seed=seed, 
        )
        rng = np.random.default_rng(noise_seed)
        return clean + noise_abs * rng.standard_normal(size=clean.shape)

    base = cfg.seed_base + TUNING_SEED_OFFSET
    hf_search = [
        sample(base + j, noise_hf_abs, noise_seed=base + 10_000 + j)
        for j in range(cfg.n_hf + 1)  # +1 held out to validate
    ]
    lf_search = [
        sample(base + 1_000 + j, noise_lf_abs, noise_seed=base + 20_000 + j)
        for j in range(cfg.n_lf)
    ]
    hf_train, hf_val = split_trajectory_list(
        hf_search, validation_fraction=1 / len(hf_search)
    )
    take_seeds = [base + 2 + t for t in range(N_TAKES)]

    grid_search = np.asarray(grid_search, dtype=float)

    # Kappa is reported, not enforced. It is defined at the true coefficients --
    # (A.1) expands the residual at Xi* -- so gating the search on it would leak
    # information no practitioner has, and would make this selection protocol
    # unreproducible on real data. Everything else here is oracle-free: the
    # metric scores held-out noisy data and the validation support follows a
    # noise ceiling computable from sigma. So kappa stays a diagnostic, and a
    # rung landing where the covariance model is invalid is a result to report
    # rather than a search to constrain.
    feature_names = get_ns_isothermal_feature_names(
        cfg, grid=grid_search, reference_trajectory=hf_val[0]
    )
    reference_coefficients = build_true_ns_isothermal_coefficients(
        feature_names, RT=cfg.RT, mu=cfg.mu
    )

    support = ns_isothermal_validation_support(
        cfg, hf_val[0], grid=grid_search, sigma=noise_hf_abs,
        candidates=grid["H_xt"], weak_seed=take_seeds[0] + 10,
        min_ceiling=MIN_CEILING, coefficients=reference_coefficients,
    )
    if verbose:
        print(support.table.to_string(index=False))
        print("\nvalidation system:", support.summary())

    val_blocks = {
        seed: build_ns_isothermal_weak_validation_blocks(
            cfg, hf_val, grid=grid_search, H_val=support.H_val,
            weak_seed=seed + 10, group_name="validation",
        )
        for seed in take_seeds
    }

    kappa_table = ns_isothermal_kappa_by_support(
        cfg, grid["H_xt"], coefficients=reference_coefficients,
        n_reference=N_KAPPA_REFERENCE,
    )
    if verbose:
        print(kappa_table.to_string(index=False))

    def score_take(candidate_cfg, weak_seed):
        coef_map = fit_ns_isothermal_multi_trajectory_coefficients(
            candidate_cfg,
            hf_trajectories=hf_train,
            lf_trajectories=lf_search,
            t_grid=t_search,
            grid=grid_search,
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
        weak_design[rung] = ns_isothermal_weak_design(rung_cfg)

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
            "derivative_order": cfg.derivative_order,
            "include_bias": cfg.include_bias,
            "p": cfg.p,
            "n_ensemble_models": cfg.n_ensemble_models,
            "N": cfg.N,
            "Nt": cfg.Nt,
            "L": cfg.L,
            "T": cfg.T,
            "dt": cfg.T / (cfg.Nt - 1),
            "mu": cfg.mu,
            "RT": cfg.RT,
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
        per_rung.results_filename = f"navierstokes_part1_{rung.lower()}_errors.csv"
        run_cfgs[rung] = per_rung

    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs per rung")
    errors = run_with_tuned_configs(
        run_ns_isothermal_multi_trajectory_gls_experiment, run_cfgs
    )
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
