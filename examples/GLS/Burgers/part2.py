# %% part2.py
"""
1D Burgers with heteroscedastic noise and weak SINDy + GLS.

- simulates 1D viscous Burgers trajectories on a periodic domain
- adds heteroscedastic Gaussian noise with variance ∝ |u_x|
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

import warnings
import os
import seaborn as sns

from utils import BurgersConfig, run_burgers_experiment
from plot_utils import bubble_hist
import pandas as pd

warnings.filterwarnings("ignore")
sns.set(context="paper", style="white", font_scale=1.1)

# %%
cfg = BurgersConfig(
        L=8.0,
        NX=256,
        t0=0.0,
        t1=10.0,
        dt=1e-2,
        nu=0.1,
        noise_level=0.25,
        poly_degree=1,
        derivative_order=2,
        H_xt=None,
        K=100,
        include_bias=True,
        stlsq_threshold=0.05,
        n_ensemble_models=100,
        n_runs=100,
        seed_base=0,
        results_dir="results",
        results_filename="burgers_weighted_errors.csv",
    )

print(
    f"Running Burgers GLS experiment with {cfg.n_runs} runs, "
    f"NX={cfg.NX}, dt={cfg.dt}, nu={cfg.nu}, "
    f"noise_level={cfg.noise_level}."
)

# %%
# df_errors, L1_errors, L0_errors = run_burgers_experiment(cfg)

# %%
# ------------------------------------------------------------------
# Reload CSV and reconstruct error dictionaries for plotting
# ------------------------------------------------------------------
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
# %%
import numpy as np

print("\nL1 errors (first few):")
for m in methods:
    print(m, np.median(L1_errors[m]))

print("\nL0 support errors (first few):")
for m in methods:
    print(m, np.median(L0_errors[m]))

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
from utils import get_burgers_gls_coefficients

C_true, C_std_single_run, C_var_single_run, C_ones_single_run = \
    get_burgers_gls_coefficients(cfg, seed=1)
    
print(C_true)
print(C_std_single_run)
print(C_var_single_run)
print(C_ones_single_run)

# %% One GLS ensemble run + coefficient posteriors (three weightings)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary, WeightedWeakPDELibrary

from utils import (
    make_space_time_grid,
    burgers_solver,
    random_initial_condition,
    build_true_burgers_coefficients,
    BurgersConfig,
)

plt.style.use("dark_background")

# -----------------------------------------------------------
# 1. Single GLS ensemble run (same settings as cfg)
# -----------------------------------------------------------
cfg_single = cfg  # reuse your BurgersConfig from above

rng = np.random.default_rng(123)
x, t = make_space_time_grid(cfg_single)
dx = x[1] - x[0]

# Clean trajectory from a random IC
u0 = random_initial_condition(rng, cfg_single)
U_clean = burgers_solver(u0, cfg_single).T       # (Nt, Nx)
U       = U_clean[:, :, None]                    # (Nt, Nx, 1)

# Heteroscedastic noise based on |u_x|
Ux = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2.0 * dx)
grad_mag = np.abs(Ux[:, :, 0])

alpha    = cfg_single.noise_level
variance = (alpha * grad_mag) ** 2
variance = np.maximum(variance, 1e-8)
std      = np.sqrt(variance)

noise   = std[:, :, None] * rng.standard_normal(size=U.shape)
U_noisy = U + noise

# Spatio-temporal grid
Xg, Tg = np.meshgrid(x, t)
XT = np.asarray([Xg, Tg]).T        # (Nt, Nx, 2)

# Base polynomial library in u
base_lib = ps.PolynomialLibrary(
    degree=cfg_single.poly_degree,
    include_bias=False,
)

tf_seed = cfg_single.seed_base + 999

# Unweighted weak library
np.random.seed(tf_seed)
weak_lib_std = WeakPDELibrary(
    function_library=base_lib,
    derivative_order=cfg_single.derivative_order,
    spatiotemporal_grid=XT,
    is_uniform=True,
    K=cfg_single.K,
    H_xt=cfg_single.H_xt,
    include_bias=cfg_single.include_bias,
)

# Variance-weighted GLS: W = V̇ᵀ Σ V̇
weights_scaled = variance / np.mean(variance)
np.random.seed(tf_seed)
weak_lib_var = WeightedWeakPDELibrary(
    function_library=base_lib,
    derivative_order=cfg_single.derivative_order,
    spatiotemporal_grid=XT,
    spatiotemporal_weights=weights_scaled,
    is_uniform=True,
    K=cfg_single.K,
    H_xt=cfg_single.H_xt,
    include_bias=cfg_single.include_bias,
)

# Ones-weighted GLS: W = V̇ᵀ V̇
np.random.seed(tf_seed)
weak_lib_ones = WeightedWeakPDELibrary(
    function_library=base_lib,
    derivative_order=cfg_single.derivative_order,
    spatiotemporal_grid=XT,
    spatiotemporal_weights=np.ones_like(variance),
    is_uniform=True,
    K=cfg_single.K,
    H_xt=cfg_single.H_xt,
    include_bias=cfg_single.include_bias,
)

def make_opt():
    return ps.EnsembleOptimizer(
        ps.STLSQ(threshold=cfg_single.stlsq_threshold),
        bagging=True,
        n_models=cfg_single.n_ensemble_models,
    )

opt_std  = make_opt()
opt_var  = make_opt()
opt_ones = make_opt()

model_std  = ps.SINDy(feature_library=weak_lib_std,  optimizer=opt_std)
model_var  = ps.SINDy(feature_library=weak_lib_var,  optimizer=opt_var)
model_ones = ps.SINDy(feature_library=weak_lib_ones, optimizer=opt_ones)

model_std.fit(U_noisy, t=t)
model_var.fit(U_noisy, t=t)
model_ones.fit(U_noisy, t=t)

# Ensemble coefficient arrays (n_models, n_features)
coefs_std  = np.array(opt_std.coef_list)[:, 0, :]
coefs_var  = np.array(opt_var.coef_list)[:, 0, :]
coefs_ones = np.array(opt_ones.coef_list)[:, 0, :]

# True coefficients for Burgers (1, 6)
C_true = build_true_burgers_coefficients(cfg_single.nu)[0]

# -----------------------------------------------------------
# 2. Coefficient posterior plots (three weightings)
# -----------------------------------------------------------
feature_labels = [r"$1$", r"$u$", r"$u_x$", r"$u_{xx}$", r"$uu_x$", r"$uu_{xx}$"]

fig, axes = plt.subplots(2, 3, figsize=(9, 4.5), dpi=200)
axes = axes.ravel()

col_var  = "#5bc0de"   # cyan: W = V̇ᵀ Σ V̇
col_ones = "#f5c542"   # yellow: W = V̇ᵀ V̇
col_std  = "#ff2f92"   # magenta: W = I

for j, ax in enumerate(axes):
    if j >= coefs_std.shape[1]:
        ax.axis("off")
        continue

    vals_std  = coefs_std[:, j]
    vals_var  = coefs_var[:, j]
    vals_ones = coefs_ones[:, j]
    c_true_j  = C_true[j]

    # Variance GLS
    if np.all(np.isfinite(vals_var)):
        sns.kdeplot(
            vals_var,
            ax=ax,
            color=col_var,
            linewidth=2.0,
            fill=True,
            alpha=0.35,
            label=r"$W=\dot V^\top\Sigma\dot V$" if j == 0 else None,
        )

    # Ones GLS
    if np.all(np.isfinite(vals_ones)):
        sns.kdeplot(
            vals_ones,
            ax=ax,
            color=col_ones,
            linewidth=2.0,
            fill=False,
            alpha=0.95,
            label=r"$W=\dot V^\top\dot V$" if j == 0 else None,
        )

    # No weighting (standard weak)
    if np.all(np.isfinite(vals_std)):
        sns.kdeplot(
            vals_std,
            ax=ax,
            color=col_std,
            linewidth=2.0,
            fill=False,
            linestyle="--",
            alpha=0.95,
            label=r"$W=I$" if j == 0 else None,
        )

    # True value
    ax.axvline(
        c_true_j,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="true" if j == 0 else None,
    )

    ax.set_title(feature_labels[j], fontsize=12)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[3].set_ylabel("Density", fontsize=12)
for ax in axes[3:]:
    ax.set_xlabel("Coefficient value", fontsize=11)

from matplotlib.lines import Line2D

axes[3].set_ylabel("Density", fontsize=12)
for ax in axes[3:]:
    ax.set_xlabel("Coefficient value", fontsize=11)

# --- clean custom legend on top ------------------------------------
legend_handles = [
    Line2D([0], [0], color=col_var,  lw=2.0,
           label=r"$W=\dot V^\top\Sigma\dot V$"),
    Line2D([0], [0], color=col_ones, lw=2.0,
           label=r"$W=\dot V^\top\dot V$"),
    Line2D([0], [0], color=col_std,  lw=2.0, ls="--",
           label=r"$W=I$"),
    Line2D([0], [0], color="white",  lw=1.5, ls="--",
           label="true"),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=4,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()

# %%# %% Forecast vs true Burgers – animation for one run (3 weightings + uncertainty)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")

# -------------------------------------------------------------------
# 1. Problem set–up
# -------------------------------------------------------------------
L   = cfg.L
NX  = cfg.NX
nu  = cfg.nu
dt  = cfg.dt

x  = np.linspace(-L, L, NX, endpoint=False)
dx = x[1] - x[0]

# New initial condition for forecast
u0_test = (
    0.8 * np.sin(2.0 * np.pi * x / L)
    + 0.2 * np.sin(4.0 * np.pi * x / L + 0.7)
)

T_forecast = 10.0
n_steps    = int(T_forecast / dt)
frame_skip = 2                 # thin out frames a bit for speed
n_frames   = n_steps // frame_skip

# -------------------------------------------------------------------
# 2. True Burgers RHS + RK4
# -------------------------------------------------------------------
def burgers_rhs_true(u):
    """Standard viscous Burgers: u_t + u u_x = nu u_xx (periodic)."""
    ux  = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
    uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
    return -u * ux + nu * uxx


def rk4_step(u, rhs, dt):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -------------------------------------------------------------------
# 3. SINDy PDE RHS from learned coefficients
# -------------------------------------------------------------------
# Feature indices must match your WeakPDELibrary order
IDX_U_UX = 4   # u u_x
IDX_U_XX = 3   # u_xx
IDX_U     = 0  # u

def burgers_rhs_sindy(u, C):
    ux  = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
    uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
    return (
        C[IDX_U_UX] * (u * ux)
        + C[IDX_U_XX] * uxx
        + C[IDX_U] * u
    )


# -------------------------------------------------------------------
# 4. Roll out: true + ensembles for the 3 weightings
# -------------------------------------------------------------------
# True trajectory frames
u_true = u0_test.copy()
U_true_frames = np.zeros((n_frames, NX))

frame_idx = 0
for n in range(n_steps):
    u_true = rk4_step(u_true, burgers_rhs_true, dt)
    if n % frame_skip == 0:
        U_true_frames[frame_idx] = u_true
        frame_idx += 1
        if frame_idx >= n_frames:
            break

def rollout_ensemble(C_list):
    """
    Roll out an ensemble of SINDy PDE models with different coefficient
    vectors C_list (n_models, n_features).

    Returns
    -------
    U_mean, U_std : (n_frames, NX)
    """
    n_models = C_list.shape[0]
    U_ens = np.zeros((n_models, n_frames, NX))

    for m in range(n_models):
        u = u0_test.copy()
        frame_idx = 0
        C = C_list[m]

        for n in range(n_steps):
            u = rk4_step(u, lambda uu, C=C: burgers_rhs_sindy(uu, C), dt)
            if n % frame_skip == 0:
                U_ens[m, frame_idx] = u
                frame_idx += 1
                if frame_idx >= n_frames:
                    break

    U_mean = np.mean(U_ens, axis=0)
    U_std  = np.std(U_ens, axis=0)
    return U_mean, U_std

# coefs_std / coefs_var / coefs_ones must be defined from the previous cell
U_std_mean,  U_std_std  = rollout_ensemble(coefs_std)
U_var_mean,  U_var_std  = rollout_ensemble(coefs_var)
U_ones_mean, U_ones_std = rollout_ensemble(coefs_ones)

# -------------------------------------------------------------------
# 5. Matplotlib animation with uncertainty bands
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3), dpi=200)

ax.set_xlim(-L, L)

all_vals = np.concatenate([
    U_true_frames.ravel(),
    (U_std_mean + U_std_std).ravel(),
    (U_std_mean - U_std_std).ravel(),
    (U_var_mean + U_var_std).ravel(),
    (U_var_mean - U_var_std).ravel(),
    (U_ones_mean + U_ones_std).ravel(),
    (U_ones_mean - U_ones_std).ravel(),
])

ymin = all_vals.min() - 0.05 * (all_vals.max() - all_vals.min())
ymax = all_vals.max() + 0.05 * (all_vals.max() - all_vals.min())
ax.set_ylim(ymin, ymax)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x,t)$")

col_std  = "tab:blue"
col_var  = "tab:green"
col_ones = "tab:orange"

(line_true,)  = ax.plot([], [], lw=2.0,  color="white",    label="True")
(line_std,)   = ax.plot([], [], '--', lw=1.4, color=col_std,   label=r"$W=I$")
(line_var,)   = ax.plot([], [], '--', lw=1.4, color=col_var,   label=r"$W=\dot{V}^\top\dot{V}$")
(line_ones,)  = ax.plot([], [], '--', lw=1.4, color=col_ones, label=r"$W=\dot{V}^\top\Sigma\dot{V}$")

# we will recreate the bands each frame (simpler; no blitting)
band_std  = None
band_var  = None
band_ones = None

ax.legend(loc="upper right", frameon=False)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

def init():
    global band_std, band_var, band_ones
    for ln in (line_true, line_std, line_var, line_ones):
        ln.set_data([], [])
    band_std = band_var = band_ones = None
    return line_true, line_std, line_var, line_ones

def update(k):
    global band_std, band_var, band_ones

    # remove old bands
    for band in (band_std, band_var, band_ones):
        if band is not None:
            band.remove()

    # set lines
    line_true.set_data(x, U_true_frames[k])
    line_std.set_data(x,  U_std_mean[k])
    line_var.set_data(x,  U_var_mean[k])
    line_ones.set_data(x, U_ones_mean[k])

    # new bands: mean ± std
    band_std = ax.fill_between(
        x,
        U_std_mean[k]  - U_std_std[k],
        U_std_mean[k]  + U_std_std[k],
        color=col_std,
        alpha=0.25,
        linewidth=0.0,
    )
    band_var = ax.fill_between(
        x,
        U_var_mean[k]  - U_var_std[k],
        U_var_mean[k]  + U_var_std[k],
        color=col_var,
        alpha=0.25,
        linewidth=0.0,
    )
    band_ones = ax.fill_between(
        x,
        U_ones_mean[k] - U_ones_std[k],
        U_ones_mean[k] + U_ones_std[k],
        color=col_ones,
        alpha=0.25,
        linewidth=0.0,
    )

    return line_true, line_std, line_var, line_ones

anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=n_frames,
    interval=40,
    blit=False,           # bands are redrawn, so no blitting
)

# optional save
anim.save("videos/burgers_forecast_weightings_uncertainty.mp4",
          writer="ffmpeg", dpi=200)


plt.tight_layout()
plt.show()
# %%
