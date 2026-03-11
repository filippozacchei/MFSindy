# %% ns_part2.py
"""
Isothermal compressible Navier–Stokes:
heteroscedastic noise + weak SINDy / GLS weighting (Part 2).

- clean reference model (weak SINDy) defines "true" coefficients
- Monte Carlo over random initial conditions:
    * No weighting (WeakPDELibrary)
    * Variance GLS (WeightedWeakPDELibrary, weights = variance_scaled)
    * Ones GLS (WeightedWeakPDELibrary, weights = 1)
- error metrics:
    * L1 error on true support
    * L0 support mismatch
- stores results in CSV + bubble-hist visualisation.
"""

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from ns_isothermal_utils import (
    NSIsothermalGLSConfig,
    run_ns_isothermal_gls_experiment,
)
from plot_utils import bubble_hist

warnings.filterwarnings("ignore")
sns.set(context="paper", style="white", font_scale=1.1)

# ---------------------------------------------------------------------
# Configuration + run experiment
# ---------------------------------------------------------------------

cfg = NSIsothermalGLSConfig(
    n_runs=25,          # lower this if the solver is too heavy
    N=64,
    Nt=500,
    L=5.0,
    T=2.5,
    mu=1.0,
    K=1000,
    RT=1.0,
    sigma0=1e-3,
    alpha=0.025,
    stlsq_threshold=0.5,
    stlsq_alpha=1e-8,
    n_ensemble_models=100,
    seed_base=1,
    results_dir="results",
    results_filename="ns_isothermal_weighted_errors.csv",
)

# %%
print(f"Running isothermal NS GLS experiment with {cfg.n_runs} runs.")
# df_errors_in_mem, L1_errors_in_mem, L0_errors_in_mem = run_ns_isothermal_gls_experiment(
#     cfg
# )

# %%
# ---------------------------------------------------------------------
# Reload CSV + reconstruct error dicts for plotting
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

# %%
# ---------------------------------------------------------------------
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
