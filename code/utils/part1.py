from utils.metrics import *
from utils.training import * 
import numpy as np
import pysindy as ps
from tqdm import tqdm
from pathlib import Path

import warnings
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
    X_lf, t_lf, X_hf, t_hf, grid, degree, threshold,
    lib=None, weights=None,
    *,  # force kwargs below
    lib_kwargs=None
):
    """
    Train LF, HF, and MF ensemble SINDy models.
    lib_kwargs: passed to WeakPDELibrary
    sindy_kwargs: passed to run_ensemble_sindy (e.g., n_models, optimizer args)
    """
    if lib is None:
        lib = ps.PolynomialLibrary(degree=degree, include_bias=False)

    lib_kwargs = {} if lib_kwargs is None else dict(lib_kwargs)
    sindy_kwargs = {} if sindy_kwargs is None else dict(sindy_kwargs)

    # Build library once
    library = ps.feature_library.WeakPDELibrary(
        lib,
        spatiotemporal_grid=grid,
        p=2,
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
    runs: int = 100,
    T: float = 0.1,
    dt: float = 1e-3,
    threshold: float = 0.5,
    degree: int = 2,
    out_dir: str = "./Results",
    C_true=None,
    seed: int = 1,
    lib=None,
    d_order: int = 0,
    *,
    # ---- kwargs passthroughs ----
    gen_kwargs: dict | None = None,       # defaults shared across all generator calls
    gen_hf_kwargs: dict | None = None,    # overrides for HF only
    gen_lf_kwargs: dict | None = None,    # overrides for LF only
    gen_test_kwargs: dict | None = None,  # overrides for test trajectory
    lib_kwargs: dict | None = None,       # passed into WeakPDELibrary
):
    """
    Generic multi-fidelity SINDy evaluation loop with kwargs passthrough.

    The kwargs logic merges dictionaries with increasing precedence:
    shared (gen_kwargs) < fidelity-specific (gen_hf_kwargs / gen_lf_kwargs) < call-time fixed args.
    """
    # Output directory
    out_path = Path(out_dir) / system_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Preallocate metric arrays
    shape = (len(n_lf_vals), len(n_hf_vals))
    score, mad, dis = _init_metrics(shape), _init_metrics(shape), _init_metrics(shape)

    # ---- Build kwargs for generator calls ----
    gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)
    gen_hf_kwargs = {} if gen_hf_kwargs is None else dict(gen_hf_kwargs)
    gen_lf_kwargs = {} if gen_lf_kwargs is None else dict(gen_lf_kwargs)
    gen_test_kwargs = {} if gen_test_kwargs is None else dict(gen_test_kwargs)

    # Clean test trajectory for R²
    test_call = {**gen_kwargs, **gen_test_kwargs}
    X_test, _, _ = generator(n_traj=5, seed=999, **test_call)
    std_scale = float(np.std(X_test))  # scale for noise

    for i, n_lf in enumerate(tqdm(n_lf_vals, desc=f"{system_name}: LF grid")):
        for j, n_hf in enumerate(n_hf_vals):

            all_runs = {
                "r2": {"hf": [], "lf": [], "mf": []},
                "mad": {"hf": [], "lf": [], "mf": []},
                "dis": {"hf": [], "lf": [], "mf": []},
            }

            for run in range(runs):
                # HF/LF-specific kwargs (on top of shared)
                hf_call = {**gen_kwargs, **gen_hf_kwargs}
                lf_call = {**gen_kwargs, **gen_lf_kwargs}

                X_hf, grid_hf, t_hf = generator(
                    n_traj=n_hf,
                    noise_level=n_lf * std_scale,
                    T=T,
                    seed=run * seed,
                    **hf_call
                )
                X_lf, _, t_lf = generator(
                    n_traj=n_lf,
                    noise_level=n_hf * std_scale,
                    T=T,
                    seed=run * seed + 100,
                    **lf_call
                )

                # Weights can still be overridden by sindy_kwargs if desired
                weights = [(1 / n_hf) ** 2] * n_hf + [(1 / n_lf) ** 2] * n_lf

                models = _train_models(
                    X_lf, t_lf, X_hf, t_hf, grid_hf, degree, threshold,
                    d_order=d_order, lib=lib, weights=weights,
                    lib_kwargs=lib_kwargs,
                )

                metrics = _evaluate_models(models, X_hf[0], dt, X_test, C_true)

                for metric_name in ("r2", "mad", "dis"):
                    for fidelity in metrics[metric_name]:
                        all_runs[metric_name][fidelity].append(metrics[metric_name][fidelity])
            agg_r2 = _aggregate_runs(all_runs["r2"], "r2")
            agg_mad = _aggregate_runs(all_runs["mad"], "mad")
            agg_dis = _aggregate_runs(all_runs["dis"], "dis")

            score["mf"][i, j] = agg_r2["mf"]
            score["lf"][i, j] = agg_r2["lf"]
            score["hf"][i, j] = agg_r2["hf"]
            score["dlf"][i, j] = agg_r2["mf"] - agg_r2["lf"]
            score["dhf"][i, j] = agg_r2["mf"] - agg_r2["hf"]

            dis["mf"][i, j] = agg_dis["mf"]
            dis["lf"][i, j] = agg_dis["lf"]
            dis["hf"][i, j] = agg_dis["hf"]
            dis["dlf"][i, j] = agg_dis["mf"] - agg_dis["lf"]
            dis["dhf"][i, j] = agg_dis["mf"] - agg_dis["hf"]

            if C_true is not None:
                mad["mf"][i, j] = agg_mad["mf"]
                mad["lf"][i, j] = agg_mad["lf"]
                mad["hf"][i, j] = agg_mad["hf"]
                mad["dlf"][i, j] = agg_mad["mf"] - agg_mad["lf"]
                mad["dhf"][i, j] = agg_mad["mf"] - agg_mad["hf"]

    np.savez_compressed(
        out_path / f"{system_name}_results.npz",
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        **{
            f"{key}_{name}": value
            for name, metric_group in zip(["score", "mad", "dis"], (score, mad, dis))
            for key, value in metric_group.items()
        },
    )

    print(f"Completed {system_name} evaluation → results saved in {out_path}")
