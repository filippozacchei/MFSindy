# %% lorenz_part2.py
"""
Lorenz system with heteroscedastic noise and weighted weak SINDy.

- simulates clean Lorenz trajectories
- adds heteroscedastic Gaussian noise (variance ∝ ‖U(t)‖)
- fits three ensemble weak SINDy models:
    * No weighting (standard WeakPDELibrary)
    * Variance GLS (WeightedWeakPDELibrary with variance weights)
    * Ones GLS (WeightedWeakPDELibrary with all-ones weights)
- runs a Monte Carlo over random initial conditions to estimate:
    * relative L1 error on coefficients
    * L0 (support) mismatch
- saves the errors to disk and visualises their distributions
  with 1D bubble histograms.
"""

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from utils import LorenzGLSConfig, run_lorenz_gls_experiment
from plot_utils import bubble_hist

warnings.filterwarnings("ignore")
sns.set(context="paper", style="white", font_scale=1.1)

# ---------------------------------------------------------------------
# Configuration and experiment
# ---------------------------------------------------------------------

cfg = LorenzGLSConfig(
    n_runs=100,
    t0=0.0,
    t1=10.0,
    dt=1e-3,
    noise_level=0.25,
    poly_degree=2,
    derivative_order=1,
    H_xt=0.01,
    p=2,
    include_bias=False,
    stlsq_threshold=0.5,
    n_ensemble_models=100,
    seed_base=0,
    results_dir="results",
    results_filename="lorenz_weighted_errors.csv",
)

print(f"Running heteroscedastic Lorenz GLS experiment with {cfg.n_runs} runs.")

# df_errors, L1_errors, L0_errors = run_lorenz_gls_experiment(cfg)

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

print("\nL1 errors (first few, reloaded from CSV):")
for m in methods:
    print(m, np.median(L1_errors[m]))

print("\nL0 support errors (first few, reloaded from CSV):")
for m in methods:
    print(m, np.median(L0_errors[m]))

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
