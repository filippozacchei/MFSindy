"""Reusable workflows for running multi-fidelity SINDy experiments."""

from pathlib import Path
import warnings

import numpy as np
import pysindy as ps
from tqdm import tqdm

from .metrics import absolute_deviation, ensemble_disagreement
from .training import copy_sindy, run_ensemble_sindy

warnings.filterwarnings("ignore")

def _init_metrics(shape):
    """Create metric containers for LF/HF/MF comparisons."""
    def zeros(): return np.full(shape, np.nan)
    return dict(
        mf=zeros(), 
        lf=zeros(), hf=zeros(),
        dlf=zeros(), dhf=zeros()
    )

def _train_models(
    X_lf, t_lf, X_hf, t_hf, grid, threshold, lib, weights=None,
    *,  # force kwargs below
    lib_kwargs=None, 
):
    """
    Train LF, HF, and MF ensemble SINDy models.
    lib_kwargs: passed to WeakPDELibrary
    """

    lib_kwargs = {} if lib_kwargs is None else dict(lib_kwargs)

    # Build library once
    library = ps.feature_library.WeakPDELibrary(
        lib,
        spatiotemporal_grid=grid,
        **lib_kwargs
    )

    model_hf, opt_hf = run_ensemble_sindy(
        X_hf, t_hf, threshold=threshold, library=library,
    )
    model_lf, opt_lf = run_ensemble_sindy(
        X_lf, t_lf, threshold=threshold, library=library,
    )

    X_mf, t_mf = X_hf + X_lf, t_hf + t_lf
    model_mf, opt_mf = run_ensemble_sindy(
        X_mf, t_mf, threshold=threshold, library=library, weights=weights,
    )

    return dict(hf=(model_hf, opt_hf), lf=(model_lf, opt_lf), mf=(model_mf, opt_mf))


def _evaluate_models(models, X_ref, dt, X_test, C_true=None):
    """Compute R², MAD, and disagreement for each model."""
    metrics = dict(r2={}, mad={}, dis={})
    for k, (model, opt) in models.items():
        eval_model, _ = copy_sindy(model, X_test, dt)
        metrics["r2"][k] = eval_model.score(X_test, t=dt)
        metrics["dis"][k] = np.median(ensemble_disagreement(opt))
        if C_true is not None:
            metrics["mad"][k] = np.median(absolute_deviation(opt.coef_.T, C_true))
    return metrics


def _aggregate_runs(metric_runs, key):
    """Aggregate medians across runs for a given metric."""
    return {k: np.median(metric_runs[k]) for k in metric_runs}


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def evaluate_mf_sindy(
    generator,
    system_name: str,
    n_lf_vals,
    n_hf_vals,
    noise_lf,
    noise_hf,
    runs: int = 100,
    dt: float = 1e-3,
    threshold: float = 0.5,
    out_dir: str = "./Results",
    C_true=None,
    seed: int = 1,
    lib=None,
    distinct_ic=True,
    verbosity=2,
    *,
    # ---- kwargs passthroughs ----
    gen_kwargs: dict | None = None,
    gen_hf_kwargs: dict | None = None,
    gen_lf_kwargs: dict | None = None,
    gen_test_kwargs: dict | None = None,
    lib_kwargs: dict | None = None,
):
    """
    Multi-fidelity SINDy evaluation loop with flexible kwargs passthrough.
    Merges kwargs in order of increasing precedence:
    shared (gen_kwargs) < fidelity-specific (gen_hf_kwargs / gen_lf_kwargs) < fixed call-time args.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    out_path = Path(out_dir) / system_name
    out_path.mkdir(parents=True, exist_ok=True)

    shape = (len(n_lf_vals), len(n_hf_vals))
    score, mad, dis = map(_init_metrics, [shape, shape, shape])

    # Handle kwargs gracefully
    gen_kwargs = gen_kwargs or {}
    gen_hf_kwargs = gen_hf_kwargs or {}
    gen_lf_kwargs = gen_lf_kwargs or {}
    gen_test_kwargs = gen_test_kwargs or {}
    lib_kwargs = lib_kwargs or {}

    # Reference trajectory (for computing R²)
    X_test, _, _ = generator(**{**gen_kwargs, **gen_test_kwargs})

    # ------------------------------------------------------------------
    # Grid evaluation
    # ------------------------------------------------------------------
    for i, n_lf in enumerate(tqdm(n_lf_vals, desc=f"{system_name}: LF grid")):
        for j, n_hf in enumerate(n_hf_vals):
            all_runs = {m: {f: [] for f in ["hf", "lf", "mf"]} for m in ["r2", "mad", "dis"]}

            # ----- Monte Carlo ensemble -----
            for run in range(runs):
                hf_call = {**gen_kwargs, **gen_hf_kwargs}
                lf_call = {**gen_kwargs, **gen_lf_kwargs}

                X_hf, grid_hf, t_hf = generator(n_traj=n_hf, seed=run * seed, **hf_call)
                n_tot_hf = len(X_hf)
                X_lf, _, t_lf = generator(n_traj=n_lf, seed=run * seed + 100*(distinct_ic), **lf_call)
                n_tot_lf = len(X_lf)
                weights = [(1 / noise_hf) ** 2] * n_tot_hf + [(1 / noise_lf) ** 2] * n_tot_lf
                models = _train_models(X_lf, t_lf, X_hf, t_hf, grid_hf,
                                       threshold, lib=lib, weights=weights,
                                       lib_kwargs=lib_kwargs)

                metrics = _evaluate_models(models, X_hf[0], dt, X_test, C_true)

                for metric_name, metric_data in metrics.items():
                    for fidelity, value in metric_data.items():
                        all_runs[metric_name][fidelity].append(value)
                        
                if verbosity >= 2:
                    r2_vals = {f: np.round(np.mean(all_runs["r2"][f]), 4) for f in ["lf", "hf", "mf"]}
                    print(f"   Run {run+1:03d}/{runs} | R²(LF)={r2_vals['lf']} "
                          f"R²(HF)={r2_vals['hf']} R²(MF)={r2_vals['mf']}")

            # ----- Aggregate statistics -----
            agg_r2 = _aggregate_runs(all_runs["r2"], "r2")
            agg_mad = _aggregate_runs(all_runs["mad"], "mad")
            agg_dis = _aggregate_runs(all_runs["dis"], "dis")

            # Store metrics
            for metric, agg in zip([score, mad, dis], [agg_r2, agg_mad, agg_dis]):
                for k in ["mf", "lf", "hf"]:
                    metric[k][i, j] = agg[k]
                metric["dlf"][i, j] = agg["mf"] - agg["lf"]
                metric["dhf"][i, j] = agg["mf"] - agg["hf"]

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    np.savez_compressed(
        out_path / f"{system_name}_results.npz",
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        **{f"{k}_{m}": v
           for m, group in zip(["score", "mad", "dis"], (score, mad, dis))
           for k, v in group.items()},
    )

    print(f"Completed {system_name} evaluation — results saved in {out_path}")
