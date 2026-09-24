"""Part II pipeline for the Pendulum benchmark, as a script.

``python main2.py`` regenerates everything ``part2.ipynb`` displays: the weak
design each weighting fits, and the Monte Carlo error table. The notebook imports
these same functions, so there is one copy of the pipeline rather than two that
drift; it can either drive them or load what this script wrote.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from mfsindy.cases.pendulum import (
    simulate_pendulum_trajectory,
    PendulumIntraTrajectoryGLSConfig,
    _build_pendulum_gls_artifacts,
    run_pendulum_intra_trajectory_gls_experiment,
)

from mfsindy.experiments import (
    TUNING_SEED_OFFSET,
    TuningArtifacts,
    WeakValidationBlock,
    build_polynomial_intra_trajectory_artifacts,
    evaluate_weak_form_models,
    fit_intra_trajectory_gls_models,
    load_tuning,
    run_with_tuned_configs,
    save_tuning,
    select_validation_support,
    split_single_trajectory,
    tune_rungs,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TUNING_PATH = RESULTS_DIR / "pendulum_part2_tuning.json"
ERRORS_PATH = RESULTS_DIR / "pendulum_part2_errors.csv"

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


def make_config(**overrides) -> PendulumIntraTrajectoryGLSConfig:
    settings = dict(
        n_runs=100,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    return PendulumIntraTrajectoryGLSConfig(**settings)


def weak_design(cfg, *, run_idx: int = 0) -> pd.DataFrame:
    """What each weighting actually fits.

    All four rungs share one set of test functions, so a difference between them
    is the weighting and nothing else. The diagonal spread is what makes the
    diagonal rung worth having: it is 1 for the unit-variance covariance by
    construction, and well above 1 wherever the noise varies along the trajectory.
    """

    artifacts = _build_pendulum_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base))
    rows = []
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
        rows.append(dict(method=method, **row))
    return pd.DataFrame(rows).set_index("method")


#: Takes per grid point: one dataset, a different test-function placement each
#: time, so a selection is not decided by where the domains happened to land.
N_TAKES = 5

#: The search trains on the first part of the trajectory and scores on the last.
VALIDATION_FRACTION = 0.2

#: Floor on the noise ceiling of the validation target. Part I uses 0.99; the
#: heteroscedastic noise here is far larger, so no support reaches that.
MIN_CEILING = 0.95


def search_grid(cfg) -> dict:
    horizon = cfg.t1 - cfg.t0
    return {
        "stlsq_threshold": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        "H_xt": [horizon / d for d in (2000.0, 1000.0, 500.0, 200.0, 100.0, 50.0)],
    }


def tuning_trajectory(cfg):
    """One noisy trajectory for the search, drawn disjoint from the evaluation."""

    rng = np.random.default_rng(cfg.seed_base + TUNING_SEED_OFFSET)
    t_eval, clean = simulate_pendulum_trajectory(
        y0=np.array([rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.5)]),
        T=cfg.t1 - cfg.t0,
        dt=cfg.dt,
        g=cfg.g,
        L=cfg.L,
        c=cfg.c,
        noise_level=0.0,
        seed=None,
    )
    variance = np.maximum((cfg.sigma0 + cfg.alpha * np.abs(clean[:, 0])) ** 2, 1e-10)
    noisy = clean + np.sqrt(variance)[:, None] * rng.standard_normal(size=clean.shape)
    return noisy, t_eval, variance


def _artifacts(cfg, data, t_grid, variance, H_xt, weak_seed):
    # K follows the support at coverage 2, the same rule the Monte Carlo builder
    # applies, here on whichever segment is passed. Left to the library it would
    # default to 100 regardless of H, and the search would then rank supports
    # under a design the run never uses.
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


def tune(cfg, *, results_dir: Path = RESULTS_DIR, verbose: bool = True) -> TuningArtifacts:
    """Select hyperparameters per rung and write the record to JSON.

    Part I holds out whole trajectories; there is only one trajectory here, so
    the last {:.0%} of it is held out instead. The validation system is built
    unweighted and scored with an unweighted R^2, so the weighting under test
    never touches the measurement -- it decides the fit, not the score.
    """.format(VALIDATION_FRACTION)

    results_dir = Path(results_dir)
    grid = search_grid(cfg)

    noisy, t_grid, variance = tuning_trajectory(cfg)
    train, validation, t_train, t_validation = split_single_trajectory(
        noisy, t_grid, validation_fraction=VALIDATION_FRACTION
    )
    n_train = train.shape[0]
    variance_train = variance[:n_train]
    variance_validation = variance[n_train - 1:]
    take_seeds = [cfg.seed_base + TUNING_SEED_OFFSET + 2 + t for t in range(N_TAKES)]

    # One fixed validation system for the whole grid, at a support chosen by the
    # noise ceiling rather than inherited from the candidate. The ceiling matters
    # more here than in Part I: the weak target carries the trajectory's own
    # heteroscedastic noise, and at the narrow end of this grid it caps the
    # attainable score near zero, where even the true coefficients score nothing.
    def build_validation(H):
        library = _artifacts(
            cfg, validation, t_validation, variance_validation, H, take_seeds[0] + 10
        ).libraries["No weighting"]
        rhs = np.asarray(library.convert_u_dot_integral(validation))
        return library, rhs

    support = select_validation_support(
        build_validation,
        candidates=grid["H_xt"],
        sigma=np.sqrt(variance_validation),
        min_ceiling=MIN_CEILING,
    )
    if verbose:
        print(support.table.to_string(index=False))
        print("\nvalidation system:", support.summary())

    val_blocks = {}
    for seed in take_seeds:
        library = _artifacts(
            cfg, validation, t_validation, variance_validation, support.H_val, seed + 10
        ).libraries["No weighting"]
        val_blocks[seed] = [
            WeakValidationBlock(
                theta=np.asarray(library.fit_transform([validation])[0]),
                rhs=np.asarray(library.convert_u_dot_integral(validation)),
            )
        ]

    def evaluate(candidate):
        frames = []
        for seed in take_seeds:
            artifacts = _artifacts(
                candidate, train, t_train, variance_train, candidate.H_xt, seed
            )
            coefficients = fit_intra_trajectory_gls_models(candidate, artifacts, METHODS)
            frames.append(evaluate_weak_form_models(coefficients, val_blocks[seed]))
        return pd.concat(frames, ignore_index=True)

    selections, score_table = tune_rungs(
        cfg,
        param_grid=grid,
        rungs=METHODS,
        metric="weak_r2",
        reducer=lambda scores: float(np.mean(np.clip(scores, -1.0, 1.0))),
        evaluate=evaluate,
    )

    design = {}
    for rung, selection in selections.items():
        rung_cfg = deepcopy(cfg)
        for name, value in selection.best_params.items():
            setattr(rung_cfg, name, value)
        design[rung] = weak_design(rung_cfg).loc[rung].to_dict()

    save_tuning(
        results_dir / TUNING_PATH.name,
        selections=selections,
        param_grid=grid,
        table=score_table,
        fixed={
            "weak_design": design,
            "n_takes": N_TAKES,
            "validation_metric": "weak_r2",
            "validation_split": {
                "fraction": VALIDATION_FRACTION,
                "note": "last fraction of the single trajectory, held out;"
                        " the validation library is unweighted, so the weighting"
                        " under test decides the fit and not the score",
            },
            "validation_support": {
                "H_val": support.H_val,
                "K": support.K,
                "ceiling": support.ceiling,
                "min_ceiling": MIN_CEILING,
            },
            "K_rule": "max(2, round((t1 - t0) / H_xt))  # coverage 2",
            "poly_degree": cfg.poly_degree,
            "p": cfg.p,
            "n_ensemble_models": cfg.n_ensemble_models,
            "t0": cfg.t0,
            "t1": cfg.t1,
            "dt": cfg.dt,
            "n_runs": cfg.n_runs,
            "seed_base": cfg.seed_base,
        },
    )
    return load_tuning(results_dir / TUNING_PATH.name)


def run_monte_carlo(cfg, selections=None, *, results_dir: Path = RESULTS_DIR,
                    verbose: bool = True) -> pd.DataFrame:
    """Run the Monte Carlo, each rung under its own tuned config when given."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"running {cfg.n_runs} Monte Carlo runs per rung")

    if selections is None:
        errors, _, _ = run_pendulum_intra_trajectory_gls_experiment(cfg)
    else:
        run_cfgs = {}
        for rung, selection in selections.items():
            per_rung = deepcopy(cfg)
            for name, value in selection.best_params.items():
                setattr(per_rung, name, value)
            per_rung.results_filename = (
                f"pendulum_part2_{rung.lower().replace(' ', '_')}_errors.csv"
            )
            run_cfgs[rung] = per_rung
        errors = run_with_tuned_configs(run_pendulum_intra_trajectory_gls_experiment, run_cfgs, source_col="method")

    errors.to_csv(results_dir / ERRORS_PATH.name, index=False)
    if verbose:
        print(errors.groupby(["method", "metric"])["value"].median().unstack().round(5))
    return errors


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_config()
    artifacts = tune(cfg)
    run_monte_carlo(cfg, artifacts.selections)


if __name__ == "__main__":
    main()
