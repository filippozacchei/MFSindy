"""Part II pipeline for the Isothermal flow benchmark, as a script.

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
TUNING_PATH = RESULTS_DIR / "navierstokes_part2_tuning.json"
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
    return NSIsothermalIntraTrajectoryGLSConfig(**settings)


def weak_design(cfg, *, run_idx: int = 0) -> pd.DataFrame:
    """What each weighting actually fits.

    All four rungs share one set of test functions, so a difference between them
    is the weighting and nothing else. The diagonal spread is what makes the
    diagonal rung worth having: it is 1 for the unit-variance covariance by
    construction, and well above 1 wherever the noise varies along the trajectory.
    """

    artifacts = _build_ns_gls_artifacts(run_idx, cfg, np.random.default_rng(cfg.seed_base))
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
N_TAKES = 3

#: The search trains on the first part of the field and scores on the last.
VALIDATION_FRACTION = 0.2

#: Floor on the noise ceiling of the validation target.
MIN_CEILING = 0.95

#: Supports offered to the validation system, as divisors of the held-out
#: window's own extents rather than the whole field's -- the window is a fifth
#: as long in time, so a support worth searching on the training part does not
#: fit inside it.
VALIDATION_SUPPORT_DIVISORS = (16.0, 8.0, 4.0, 2.5)


def search_grid(cfg) -> dict:
    h_s = [cfg.L / d for d in (20.0, 10.0, 5.0)]
    h_t = [cfg.T / d for d in (80.0, 40.0, 20.0)]
    return {
        "stlsq_threshold": [0.1, 0.2, 0.5, 1.0],
        "H_xt": [[hs, hs, ht] for hs in h_s for ht in h_t],
    }


def _truth(cfg):
    """The grid, feature library and true coefficients the runner builds.

    Reproduced here rather than imported because the runner assembles them
    inside itself; the tuning path needs the same three, and they are fixed by
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


def tuning_trajectory(cfg):
    """One noisy flow for the search, drawn disjoint from the evaluation.

    Same solver and same noise model as the Monte Carlo -- variance
    ``(sigma0 + alpha |du/dt|)^2`` -- at a seed offset by TUNING_SEED_OFFSET.
    """

    U_clean, t, grid = generate_isothermal_ns_dataset(
        N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT,
        seed=cfg.seed_base + TUNING_SEED_OFFSET,
    )
    noisy, variance = add_heteroscedastic_noise_temporal_derivative(
        U_clean, t, sigma0=cfg.sigma0, alpha=cfg.alpha,
        rng=np.random.default_rng(cfg.seed_base + TUNING_SEED_OFFSET),
    )
    return noisy, np.asarray(grid, dtype=float), t, variance


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
    """Kappa per candidate fitting support, on the tuning flow.

    Dimensionless and sigma-weighted, as for Burgers. A diagnostic, not a
    constraint: a support where kappa is large is one where the weighting should
    help least, which is a result rather than a reason to drop the point.
    """

    _, base_library, coefficients = _truth(cfg)
    noisy, grid, t_grid, variance = tuning_trajectory(cfg)
    train, _ = split_pde_field_in_time(
        noisy, grid, t_grid, variance=variance,
        validation_fraction=VALIDATION_FRACTION,
    )
    sigma = np.sqrt(train["variance"])
    seed = cfg.seed_base + TUNING_SEED_OFFSET + 2

    rows = []
    for H in search_grid(cfg)["H_xt"]:
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


def tune(cfg, *, results_dir: Path = RESULTS_DIR, verbose: bool = True) -> TuningArtifacts:
    """Select hyperparameters per rung and write the record to JSON.

    Part I holds out whole trajectories; there is one flow here, so its last
    {:.0%} in time is held out instead. The validation system is built unweighted
    and scored with an unweighted R^2, so the weighting under test never touches
    the measurement -- it decides the fit, not the score.
    """.format(VALIDATION_FRACTION)

    results_dir = Path(results_dir)
    grid = search_grid(cfg)
    _, base_library, coefficients = _truth(cfg)

    noisy, xt_grid, t_grid, variance = tuning_trajectory(cfg)
    train, validation = split_pde_field_in_time(
        noisy, xt_grid, t_grid, variance=variance,
        validation_fraction=VALIDATION_FRACTION,
    )
    take_seeds = [cfg.seed_base + TUNING_SEED_OFFSET + 2 + t for t in range(N_TAKES)]

    def build_validation(H):
        library = _artifacts(
            cfg, validation, H, take_seeds[0] + 10,
            base_library=base_library, coefficients=coefficients,
        ).libraries["No weighting"]
        # fitted here because the kappa diagnostic needs the library's feature
        # layout, which only exists once it has seen the data
        library.fit_transform([validation["data"]])
        return library, np.asarray(library.convert_u_dot_integral(validation["data"]))

    extents = [
        float(np.ptp(validation["grid"][..., axis]))
        for axis in range(validation["grid"].shape[-1])
    ]
    sigma_validation = np.sqrt(validation["variance"])
    # Kappa is reported against each candidate but does not bind here: it governs
    # the covariance model, and this system invokes none.
    support = select_validation_support(
        build_validation,
        candidates=[[e / d for e in extents] for d in VALIDATION_SUPPORT_DIVISORS],
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
        library = _artifacts(
            cfg, validation, support.H_val, seed + 10,
            base_library=base_library, coefficients=coefficients,
        ).libraries["No weighting"]
        val_blocks[seed] = [
            WeakValidationBlock(
                theta=np.asarray(library.fit_transform([validation["data"]])[0]),
                rhs=np.asarray(library.convert_u_dot_integral(validation["data"])),
            )
        ]

    def evaluate(candidate):
        frames = []
        for seed in take_seeds:
            artifacts = _artifacts(
                candidate, train, candidate.H_xt, seed,
                base_library=base_library, coefficients=coefficients,
            )
            fitted = fit_intra_trajectory_gls_models(candidate, artifacts, METHODS)
            frames.append(evaluate_weak_form_models(fitted, val_blocks[seed]))
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
        (float(row.H_s), float(row.H_t)): float(row.kappa_median)
        for row in kappa_table.itertuples()
    }
    kappa_at_selection = {
        rung: kappa_at_support[
            (float(selection.best_params["H_xt"][0]), float(selection.best_params["H_xt"][2]))
        ]
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
                "note": "last fraction of the single flow, held out; the"
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
            "derivative_order": cfg.derivative_order,
            "p": cfg.p,
            "n_ensemble_models": cfg.n_ensemble_models,
            "mu": cfg.mu,
            "RT": cfg.RT,
            "N": cfg.N,
            "Nt": cfg.Nt,
            "L": cfg.L,
            "T": cfg.T,
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
        errors, _, _ = run_ns_isothermal_intra_trajectory_gls_experiment(cfg)
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
                f"navierstokes_part2_{rung.lower().replace(' ', '_')}_errors.csv"
            )
            run_cfgs[rung] = per_rung
        errors = run_with_tuned_configs(
            run_ns_isothermal_intra_trajectory_gls_experiment,
            run_cfgs,
            source_col="method",
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
