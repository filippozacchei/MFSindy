# %% pendulum_part2.py
"""
Single pendulum with heteroscedastic noise and weak SINDy + GLS.

- simulates damped pendulum trajectories
- adds heteroscedastic Gaussian noise (variance depending on |omega(t)|)
- fits three ensemble weak SINDy models:
    * No weighting (standard WeakPDELibrary)
    * Variance GLS (WeightedWeakPDELibrary with variance weights)
    * Ones GLS (WeightedWeakPDELibrary with all-ones weights)
- runs a Monte Carlo over random initial conditions to estimate:
    * L1 error on coefficients (on true support)
    * L0 (support) mismatch
- saves the errors to disk and visualises their distributions
  with 1D bubble histograms.
"""

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from pendulum_utils import PendulumGLSConfig, run_pendulum_gls_experiment
from plot_utils import bubble_hist

warnings.filterwarnings("ignore")
sns.set(context="paper", style="white", font_scale=1.1)

# ---------------------------------------------------------------------
# Configuration and experiment
# ---------------------------------------------------------------------

cfg = PendulumGLSConfig(
    n_runs=100,
    t0=0.0,
    t1=10.0,
    dt=1e-3,
    g=9.81,
    L=1.0,
    c=0.5,
    sigma0=0.0,
    alpha=0.25,
    poly_degree=1,
    derivative_order=1,
    H_xt=0.1,
    K=500,
    p=2,
    include_bias=False,
    stlsq_threshold=0.05,
    n_ensemble_models=100,
    seed_base=0,
    results_dir="results",
    results_filename="pendulum_weighted_errors.csv",
)

print(f"Running heteroscedastic pendulum GLS experiment with {cfg.n_runs} runs.")

# df_errors, L1_errors_in_mem, L0_errors_in_mem = run_pendulum_gls_experiment(cfg)

# ---------------------------------------------------------------------
# Reload CSV and reconstruct error dicts for plotting
# ---------------------------------------------------------------------

errors_path = os.path.join(cfg.results_dir, cfg.results_filename)
df_errors = pd.read_csv(errors_path)

methods = ["No weighting", "Variance GLS", "Ones GLS"]

L1_errors = {
    m: df_errors[
        (df_errors["method"] == m) & (df_errors["metric"] == "L1")
    ]["value"].to_numpy()
    for m in methods
}
L0_errors = {
    m: df_errors[
        (df_errors["method"] == m) & (df_errors["metric"] == "L0")
    ]["value"].to_numpy()
    for m in methods
}

print("\nMedian L1 errors (reloaded from CSV):")
for m in methods:
    print(f"{m:14s} : {np.median(L1_errors[m]):.3e}")

print("\nMedian L0 support errors (reloaded from CSV):")
for m in methods:
    print(f"{m:14s} : {np.median(L0_errors[m]):.3e}")

# %%---------------------------------------------------------------------
# Bubble-hist plots for L1 and L0
# ---------------------------------------------------------------------

method_colors = {
    "No weighting": "tab:blue",
    "Variance GLS": "tab:green",
    "Ones GLS": "tab:orange",
}
methods = ["Variance GLS", "Ones GLS", "No weighting"]
labels = [r"$\dot{V}^\top\Sigma\dot{V}$",r"$\dot{V}^\top\dot{V}$","I"]

bubble_hist(
    errors_dict=L1_errors,
    title=f"Isothermal NS (heteroscedastic): L1 error ({cfg.n_runs} runs)",
    xlabel=r"$L_1$ error",
    n_bins=20,
    models_order=methods,
    colors=method_colors,
    labels=labels
)

bubble_hist(
    errors_dict=L0_errors,
    title=f"Isothermal NS (heteroscedastic): $L_0$ support error ({cfg.n_runs} runs)",
    xlabel=r"$L_0$ error",
    n_bins=10,
    models_order=methods,
    colors=method_colors,
    labels=labels
)
# %%
