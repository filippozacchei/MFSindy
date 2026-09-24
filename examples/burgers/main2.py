"""Part II pipeline for the Burgers benchmark, as a script.

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

from mfsindy.cases.burgers import (
    BurgersConfig,
    _build_burgers_gls_artifacts,
    build_true_burgers_coefficients,
    burgers_solver,
    make_space_time_grid,
    random_initial_condition,
    run_burgers_experiment,
)

from mfsindy.weighted_weak_pde_library import (
    pde_sensitivity_fields,
    pde_weak_validity_ratio,
)

from mfsindy.experiments import (
    TUNING_SEED_OFFSET,
    TuningArtifacts,
    WeakValidationBlock,
    build_pde_intra_trajectory_artifacts,
    evaluate_weak_form_models,
    fit_intra_trajectory_gls_models,
    load_tuning,
    run_with_tuned_configs,
    save_tuning,
    select_validation_support,
    split_pde_field_in_time,
    tune_rungs,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TUNING_PATH = RESULTS_DIR / "burgers_part2_tuning.json"
ERRORS_PATH = RESULTS_DIR / "burgers_part2_errors.csv"

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


def make_config(**overrides) -> BurgersConfig:
    settings = dict(
        n_runs=100,
        results_dir=str(RESULTS_DIR),
        results_filename=ERRORS_PATH.name,
    )
    settings.update(overrides)
    return BurgersConfig(**settings)


def weak_design(cfg, *, run_idx: int = 0) -> pd.DataFrame:
    """What each weighting actually fits.

    All four rungs share one set of test functions, so a difference between them
    is the weighting and nothing else. The diagonal spread is what makes the
    diagonal rung worth having: it is 1 for the unit-variance covariance by
    construction, and well above 1 wherever the noise varies along the trajectory.
    """

    artifacts = _build_burgers_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base))
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

#: The search trains on the first part of the field and scores on the last.
VALIDATION_FRACTION = 0.2

#: Floor on the noise ceiling of the validation target. Part I uses 0.99; the
#: heteroscedastic noise here is far larger, so that is out of reach.
MIN_CEILING = 0.95

#: Supports offered to the validation system, as divisors of the held-out
#: window's own extents rather than the whole field's. The validation system is
#: the measuring instrument, not a candidate, so it is sized by the data it is
#: built on -- the window is a fifth as long in time, and a support wide enough
#: to be worth searching on the training part does not fit inside it.
#:
#: Space and time are divided separately. Measured over the held-out window, the
#: ceiling here is set almost entirely by the temporal width -- at H_t = 0.8 it
#: is 0.96 to 0.99 whatever H_x does, and at H_t = 0.1 it never exceeds 0.28 --
#: which is the opposite of the flow, where the spatial width governs. So no
#: single divisor serves both cases, and tying the two together here left one
#: qualifying candidate with three rows: a validation system that cannot rank
#: anything.
VALIDATION_SPACE_DIVISORS = (20.0, 10.0, 5.0)
VALIDATION_TIME_DIVISORS = (10.0, 5.0, 2.5)


def search_grid(cfg) -> dict:
    h_x = [cfg.L / d for d in (16.0, 8.0, 4.0)]
    h_t = [(cfg.t1 - cfg.t0) / d for d in (20.0, 10.0, 5.0)]
    return {
        "stlsq_threshold": [0.01, 0.02, 0.05, 0.1, 0.2],
        "H_xt": [[hx, ht] for hx in h_x for ht in h_t],
    }


def tuning_trajectory(cfg):
    """One noisy field for the search, drawn disjoint from the evaluation.

    Same solver and same noise model as the Monte Carlo -- variance
    ``(alpha |u_x|)^2`` -- at a seed offset by TUNING_SEED_OFFSET, so nothing the
    search sees is reused when the rungs are scored.
    """

    rng = np.random.default_rng(cfg.seed_base + TUNING_SEED_OFFSET)
    x, t = make_space_time_grid(cfg)
    dx = x[1] - x[0]

    clean = burgers_solver(random_initial_condition(rng, cfg), cfg).T[:, :, None]
    u_x = (np.roll(clean, -1, axis=0) - np.roll(clean, 1, axis=0)) / (2.0 * dx)
    variance = np.maximum((cfg.noise_level * np.abs(u_x[:, :, 0])) ** 2, 1e-8)
    noisy = clean + np.sqrt(variance)[:, :, None] * rng.standard_normal(size=clean.shape)

    X, T = np.meshgrid(x, t)
    return noisy, np.asarray([X, T]).T, t, variance


def _artifacts(cfg, window, H_xt, weak_seed):
    """The four weightings on one window of the field, K following the support."""

    return build_pde_intra_trajectory_artifacts(
        window["data"],
        window["grid"],
        t_grid=window["t_grid"],
        variance=window["variance"] / np.mean(window["variance"]),
        poly_degree=cfg.poly_degree,
        derivative_order=cfg.derivative_order,
        include_bias=cfg.include_bias,
        function_include_bias=False,
        H_xt=H_xt,
        p=cfg.p,
        weak_seed=weak_seed,
        true_coefficients=build_true_burgers_coefficients(cfg.nu),
    )


def kappa_by_support(cfg, *, verbose: bool = True) -> pd.DataFrame:
    """Kappa per candidate fitting support, on the tuning field.

    Dimensionless, so the numbers are comparable across supports and against the
    ODE cases; the scaling form h_t^2 h_x^-2m carries units and is not. Sigma is
    carried because the noise here varies over the field and so does not cancel
    between the two terms of the ratio.

    A diagnostic, not a constraint: kappa governs the covariance model, and a
    support where it is large is one where the weighting should help least,
    which is a result rather than a reason to drop the point.
    """

    noisy, grid, t_grid, variance = tuning_trajectory(cfg)
    train, _ = split_pde_field_in_time(
        noisy, grid, t_grid, variance=variance,
        validation_fraction=VALIDATION_FRACTION,
    )
    sigma = np.sqrt(train["variance"])
    coefficients = build_true_burgers_coefficients(cfg.nu)
    seed = cfg.seed_base + TUNING_SEED_OFFSET + 2

    rows = []
    for H in search_grid(cfg)["H_xt"]:
        library = _artifacts(cfg, train, H, seed).libraries["No weighting"]
        library.fit_transform([train["data"]])
        ratios = pde_weak_validity_ratio(
            library,
            pde_sensitivity_fields(library, train["data"], coefficients),
            sigma=sigma,
        )
        rows.append(
            dict(
                H_x=H[0],
                H_t=H[1],
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


def tune(cfg, *, results_dir: Path = RESULTS_DIR, verbose: bool = True) -> TuningArtifacts:
    """Select hyperparameters per rung and write the record to JSON.

    Part I holds out whole trajectories; there is one field here, so its last
    {:.0%} in time is held out instead. The validation system is built unweighted
    and scored with an unweighted R^2, so the weighting under test never touches
    the measurement -- it decides the fit, not the score.
    """.format(VALIDATION_FRACTION)

    results_dir = Path(results_dir)
    grid = search_grid(cfg)

    noisy, xt_grid, t_grid, variance = tuning_trajectory(cfg)
    train, validation = split_pde_field_in_time(
        noisy, xt_grid, t_grid, variance=variance,
        validation_fraction=VALIDATION_FRACTION,
    )
    take_seeds = [cfg.seed_base + TUNING_SEED_OFFSET + 2 + t for t in range(N_TAKES)]

    # One fixed validation system for the whole grid, at a support chosen by the
    # noise ceiling rather than inherited from the candidate. Letting it follow
    # the candidate would give every grid point its own target, which is not a
    # ranking.
    def build_validation(H):
        library = _artifacts(cfg, validation, H, take_seeds[0] + 10).libraries["No weighting"]
        # fitted here because the kappa diagnostic needs the library's feature
        # layout, which only exists once it has seen the data
        library.fit_transform([validation["data"]])
        rhs = np.asarray(library.convert_u_dot_integral(validation["data"]))
        return library, rhs

    extents = [
        float(np.ptp(validation["grid"][..., axis]))
        for axis in range(validation["grid"].shape[-1])
    ]
    sigma_validation = np.sqrt(validation["variance"])
    # Kappa is reported against each candidate but does not bind here: it governs
    # the covariance model, and this system invokes none.
    coefficients = build_true_burgers_coefficients(cfg.nu)
    support = select_validation_support(
        build_validation,
        candidates=[
            [extents[0] / dx, extents[1] / dt]
            for dx in VALIDATION_SPACE_DIVISORS
            for dt in VALIDATION_TIME_DIVISORS
        ],
        sigma=sigma_validation,
        kappa_fn=lambda library, _H: float(
            np.median(
                pde_weak_validity_ratio(
                    library,
                    pde_sensitivity_fields(library, validation["data"], coefficients),
                    sigma=sigma_validation,
                )
            )
        ),
        min_ceiling=MIN_CEILING,
    )
    if verbose:
        print(support.table.to_string(index=False))
        print("\nvalidation system:", support.summary())

    val_blocks = {}
    for seed in take_seeds:
        library = _artifacts(cfg, validation, support.H_val, seed + 10).libraries["No weighting"]
        val_blocks[seed] = [
            WeakValidationBlock(
                theta=np.asarray(library.fit_transform([validation["data"]])[0]),
                rhs=np.asarray(library.convert_u_dot_integral(validation["data"])),
            )
        ]

    def evaluate(candidate):
        frames = []
        for seed in take_seeds:
            artifacts = _artifacts(candidate, train, candidate.H_xt, seed)
            coefficients = fit_intra_trajectory_gls_models(candidate, artifacts, METHODS)
            frames.append(evaluate_weak_form_models(coefficients, val_blocks[seed]))
        return pd.concat(frames, ignore_index=True)

    kappa_table = kappa_by_support(cfg, verbose=verbose)

    selections, score_table = tune_rungs(
        cfg,
        param_grid=grid,
        rungs=METHODS,
        metric="weak_r2",
        reducer=lambda scores: float(np.mean(np.clip(scores, -1.0, 1.0))),
        evaluate=evaluate,
    )

    kappa_at_support = {
        (float(row.H_x), float(row.H_t)): float(row.kappa_median)
        for row in kappa_table.itertuples()
    }
    kappa_at_selection = {
        rung: kappa_at_support[tuple(float(h) for h in selection.best_params["H_xt"])]
        for rung, selection in selections.items()
    }
    if verbose:
        print("\nkappa at each selection: "
              + ", ".join(f"{r}={k:.3g}" for r, k in kappa_at_selection.items()))

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
        kappa_table=kappa_table,
        fixed={
            "weak_design": design,
            "n_takes": N_TAKES,
            "validation_metric": "weak_r2",
            "validation_split": {
                "fraction": VALIDATION_FRACTION,
                "axis": "time",
                "note": "last fraction of the single field, held out; the"
                        " validation library is unweighted, so the weighting"
                        " under test decides the fit and not the score",
            },
            "validation_support": {
                "H_val": list(np.atleast_1d(support.H_val)),
                "K": support.K,
                "ceiling": support.ceiling,
                "min_ceiling": MIN_CEILING,
            },
            "kappa_diagnostic": {
                "definition": "dimensionless PDE ratio, sigma-weighted, on the"
                              " training window",
                "statistic": "kappa_median (median over test functions);"
                             " reported, never used to restrict the grid",
                "kappa_at_selection": kappa_at_selection,
            },
            "K_rule": "max(2, round(2 * prod(extents) / prod(2H)))  # coverage 2",
            "poly_degree": cfg.poly_degree,
            "derivative_order": cfg.derivative_order,
            "p": cfg.p,
            "n_ensemble_models": cfg.n_ensemble_models,
            "nu": cfg.nu,
            "L": cfg.L,
            "NX": cfg.NX,
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
        errors, _, _ = run_burgers_experiment(cfg)
    else:
        run_cfgs = {}
        for rung, selection in selections.items():
            per_rung = deepcopy(cfg)
            for name, value in selection.best_params.items():
                setattr(per_rung, name, value)
            # results_dir is where this run's output goes, per-rung CSVs included:
            # without this a scratch run writes over the committed results.
            per_rung.results_dir = str(results_dir)
            per_rung.results_filename = (
                f"burgers_part2_{rung.lower().replace(' ', '_')}_errors.csv"
            )
            run_cfgs[rung] = per_rung
        errors = run_with_tuned_configs(
            run_burgers_experiment, run_cfgs, source_col="method"
        )

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
