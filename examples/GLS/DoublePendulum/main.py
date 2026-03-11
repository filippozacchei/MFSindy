# %% [markdown]
# # Nonlinear Single Pendulum — Weak SINDy + GLS Weighting (Exact Model, Degree 3 + sinθ)

# We consider a planar single pendulum with viscous damping. The state is

# \[
# y(t) = [\theta(t),\,\omega(t)]^\top,
# \]

# where:

# - $\theta$: angle of the pendulum w.r.t. the vertical,
# - $\omega$: angular velocity.

# The equations of motion are

# \[
# \dot \theta = \omega,\qquad
# \dot \omega = -\frac{g}{L}\sin(\theta) - c\,\omega,
# \]

# with parameters \(g>0\) (gravity), \(L>0\) (length), and damping \(c>0\).

# We choose a small **Fourier-like feature library** that contains
# \(\theta\), \(\omega\), and \(\sin\theta\). In this basis, the true model
# has *exactly* the form

# \[
# \begin{aligned}
# \dot\theta &= 0\cdot \theta + 1\cdot \omega + 0\cdot \sin\theta, \\
# \dot\omega &= 0\cdot \theta - c\cdot \omega - \frac{g}{L}\sin\theta,
# \end{aligned}
# \]

# so that we can directly measure how well weak SINDy (with different
# GLS weightings) recovers these coefficients.

# In this notebook:

# 1. Define the nonlinear pendulum dynamics and simulate clean trajectories.
# 2. Define a **compact custom feature library**:
#    \([\theta,\;\omega,\;\sin\theta]\).
# 3. Introduce **heteroscedastic noise** whose variance grows with
#    \(|\omega(t)|\).
# 4. Fit three weak SINDy models on noisy data:
#    - **No weighting** (standard weak SINDy),
#    - **Variance-based GLS** (weights $\propto \sigma^2(t)$),
#    - **Unit-weight GLS** (`weights = 1`).
# 5. Compute:
#    - mean relative \(L_1\) coefficient error vs the **true** coefficients,
#    - \(L_0\) support mismatch.
# 6. Run a **Monte Carlo** experiment over random initial conditions and
#    summarize error distributions.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary, WeightedWeakPDELibrary

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# %% [markdown]
# ## 1. Single Pendulum Dynamics and Integrator
#
# The ODE system is
#
# \[
# y = [\theta,\omega]^\top,\quad
# \dot y = f(y),
# \]
#
# integrated with an explicit RK4 scheme.

# %%
# -------------------------------
# Nonlinear single pendulum
# -------------------------------
def pendulum_rhs(y, g=9.81, L=1.0, c=0.05):
    """
    Time derivative for a planar single pendulum with viscous damping.

    y = [theta, omega].
    """
    theta, omega = y
    dtheta = omega
    domega = -(g / L) * np.sin(theta) - c * omega
    return np.array([dtheta, domega])


def rk4_step(y, h):
    k1 = pendulum_rhs(y)
    k2 = pendulum_rhs(y + 0.5 * h * k1)
    k3 = pendulum_rhs(y + 0.5 * h * k2)
    k4 = pendulum_rhs(y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(y0, T=10.0, dt=1e-3):
    """Integrate the single pendulum from y0 up to time T with step dt."""
    n = int(T / dt)
    Y = np.zeros((n, len(y0)))
    Y[0] = y0
    for i in range(1, n):
        Y[i] = rk4_step(Y[i - 1], dt)
    t = np.arange(n) * dt
    return t, Y


# Parameters (for true model)
g = 9.81
L = 1.0
c = 0.05

# %% [markdown]
# ## 2. Clean Reference Trajectory

# We generate a moderately nonlinear clean trajectory and visualize it.

# %%
# Reference trajectory parameters
dt_ref = 1e-3
T_ref = 20.0

# Moderately nonlinear initial condition
y0_ref = np.array([1.5, 0.0])   # [theta0, omega0]

t_ref, Y_ref = simulate(y0_ref, T=T_ref, dt=dt_ref)
print("Reference trajectory shape:", Y_ref.shape)

# Plot reference trajectory (theta, omega)
fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True, dpi=150)

ax[0].plot(t_ref, Y_ref[:, 0], "k", lw=1.2)
ax[0].set_ylabel(r"$\theta(t)$")
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)

ax[1].plot(t_ref, Y_ref[:, 1], "k", lw=1.2)
ax[1].set_xlabel("t [s]")
ax[1].set_ylabel(r"$\omega(t)$")
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# Phase portrait
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
ax.plot(Y_ref[:, 0], Y_ref[:, 1], "k", lw=1.0, alpha=0.9)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\omega$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Pendulum phase portrait (clean)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Custom Feature Library with sin(θ)
#
# We build a **compact custom library** with features
#
# \[
# \Theta(y) = [\theta,\;\omega,\;\sin\theta].
# \]
#
# In this basis, the **true coefficients** are
#
# \[
# \Xi_{\text{true}} =
# \begin{bmatrix}
# 0 & 1 & 0 \\
# 0 & -c & -g/L
# \end{bmatrix}.
# \]

# %%
# Custom feature library: [theta, omega, sin(theta)]
def theta_fun(X):
    return X[:, 0:1]          # column vector


def omega_fun(X):
    return X[:, 1:2]


def sin_theta_fun(X):
    return np.sin(X[:, 0:1])


library_functions = [theta_fun, omega_fun, sin_theta_fun]
library_names = [r"theta", r"omega", r"sin(theta)"]

custom_library = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_names,
)

# True coefficients in this basis:
# ordering: [theta, omega, sin(theta)]
true_coef = np.array([
    [0.0,     1.0,    0.0],        # theta_dot
    [0.0,    -c,   -g / L],        # omega_dot
])  # shape (2, 3)

C_true_flat = true_coef.reshape(-1)
print("true_coef shape:", true_coef.shape)

# %% [markdown]
# ## 4. Heteroscedastic Noise Model for Identification
#
# For identification experiments, we use shorter trajectories and add
# **state-dependent noise**.
#
# Let
# \[
# \kappa(t) = |\omega(t)|
# \]
# be the instantaneous angular velocity magnitude. We define the variance
#
# \[
# \sigma^2(t) = \bigl(\sigma_0 + \alpha\,\kappa(t)\bigr)^2,
# \]
#
# and noisy observations
#
# \[
# Y_{\text{noisy}}(t) = Y(t) + \varepsilon(t),\qquad
# \varepsilon(t) \sim \mathcal N(0,\,\sigma^2(t) I_2).
# \]
#
# This gives a scalar variance field in time, which we feed to
# `WeightedWeakPDELibrary` as `spatiotemporal_weights`.

# %%
# Shorter trajectory for a *single* identification experiment
dt_id = 1e-3
T_id = 5.0
t_id, Y_id_clean = simulate(y0_ref, T=T_id, dt=dt_id)
print("Identification trajectory shape:", Y_id_clean.shape)

# Heteroscedastic variance based on |omega|
omega_mag = np.abs(Y_id_clean[:, 1])   # (T_id,)
sigma0 = 1e-3
alpha_noise = 0.25

variance_id = (sigma0 + alpha_noise * omega_mag)**2  # (T_id,)
variance_id = np.maximum(variance_id, 1e-10)
std_id = np.sqrt(variance_id)

rng = np.random.default_rng(123)
noise_id = std_id[:, None] * rng.standard_normal(size=Y_id_clean.shape)
Y_id_noisy = Y_id_clean + noise_id

print("Noise std range (single traj):", std_id.min(), "to", std_id.max())

# Plot θ, ω with overlaid σ(t)
fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
ax.plot(t_id, Y_id_noisy[:, 0], lw=1.0, label=r"$\theta$ noisy")
ax.plot(t_id, Y_id_noisy[:, 1], lw=1.0, label=r"$\omega$ noisy")
ax.plot(t_id, std_id, lw=1.6, color="firebrick", label=r"$\sigma(t)$")

ax.set_xlabel("t [s]")
ax.set_ylabel(r"state / $\sigma(t)$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Weak SINDy with Temporal Weighting (Single Trajectory)

# We now construct three weak libraries for the *noisy* identification
# trajectory:

# - **No weighting**: standard weak SINDy.
# - **Variance GLS**: temporal weights equal to the variance field
#   $\sigma^2(t)$.
# - **Unit GLS**: temporal weights are identically one.

# Each model is fit with an **ensemble optimizer** and we compare each
# ensemble member against the **true coefficients** in our
# \([\theta,\omega,\sin\theta]\) basis.

# Errors:

# - **Relative \(L_1\) error** on nonzero entries of $\Xi_{\text{true}}$,
# - **\(L_0\) support error**: number of indices where the algebraic support
#   differs between estimate and truth.

# %%
# Helper: coefficient error metrics
def coeff_errors(C_est, C_true, tol_support=1e-6, tol_rel=1e-8):
    """
    Relative L1 error + L0 support mismatch.

    C_est, C_true : 1D arrays.
    """
    C_est = np.asarray(C_est).ravel()
    C_true = np.asarray(C_true).ravel()

    # L0 support mismatch
    supp_true = np.abs(C_true) > tol_support
    supp_est  = np.abs(C_est)  > tol_support
    l0_err = np.count_nonzero(supp_true ^ supp_est)

    # Relative L1 on nonzero true coefficients
    if np.any(supp_true):
        C_true_nz = C_true[supp_true]
        C_est_nz  = C_est[supp_true]
        denom = np.maximum(np.abs(C_true_nz), tol_rel)
        rel_err = np.abs(C_est_nz - C_true_nz) / denom
        l1_rel = np.mean(rel_err)
    else:
        l1_rel = 0.0

    return l1_rel, l0_err


# %%
# Weak and weighted weak libraries for the identification trajectory
XT_id = t_id[:, None]

weak_lib = WeakPDELibrary(
    function_library=custom_library,
    derivative_order=1,
    spatiotemporal_grid=XT_id,
    is_uniform=True,
    include_bias=False,
)

weighted_weak_lib_var = WeightedWeakPDELibrary(
    function_library=custom_library,
    derivative_order=1,
    spatiotemporal_grid=XT_id,
    spatiotemporal_weights=variance_id,  # (T_id,)
    is_uniform=True,
    include_bias=False,
)

weighted_weak_lib_ones = WeightedWeakPDELibrary(
    function_library=custom_library,
    derivative_order=1,
    spatiotemporal_grid=XT_id,
    spatiotemporal_weights=np.ones_like(variance_id),
    is_uniform=True,
    include_bias=False,
)

# Ensemble optimizers
opt_std = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                               bagging=True, n_models=40)
opt_var = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                               bagging=True, n_models=40)
opt_ones = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                bagging=True, n_models=40)

model_std = ps.SINDy(feature_library=weak_lib, optimizer=opt_std)
model_var = ps.SINDy(feature_library=weighted_weak_lib_var, optimizer=opt_var)
model_ones = ps.SINDy(feature_library=weighted_weak_lib_ones, optimizer=opt_ones)

model_std.fit(Y_id_noisy, t=t_id)
model_var.fit(Y_id_noisy, t=t_id)
model_ones.fit(Y_id_noisy, t=t_id)

print("\n===== Weak SINDy (no weighting) =====")
model_std.print()
print("\n===== Weighted Weak SINDy (variance GLS) =====")
model_var.print()
print("\n===== Weighted Weak SINDy (weights = 1 GLS) =====")
model_ones.print()

# %%
# Stack ensemble coefficients: (E, 2, n_terms)
coef_std  = np.stack(opt_std.coef_list,  axis=0)
coef_var  = np.stack(opt_var.coef_list,  axis=0)
coef_ones = np.stack(opt_ones.coef_list, axis=0)

print("coef_std shape:  ", coef_std.shape)
print("coef_var shape:  ", coef_var.shape)
print("coef_ones shape: ", coef_ones.shape)

E, n_states, n_terms = coef_std.shape

rel_L1_std, rel_L1_var, rel_L1_ones = [], [], []
L0_std, L0_var, L0_ones = [], [], []

for e in range(E):
    C_std_e  = coef_std[e].reshape(-1)
    C_var_e  = coef_var[e].reshape(-1)
    C_ones_e = coef_ones[e].reshape(-1)

    l1_s, l0_s = coeff_errors(C_std_e,  C_true_flat)
    l1_v, l0_v = coeff_errors(C_var_e,  C_true_flat)
    l1_o, l0_o = coeff_errors(C_ones_e, C_true_flat)

    rel_L1_std.append(l1_s);   L0_std.append(l0_s)
    rel_L1_var.append(l1_v);   L0_var.append(l0_v)
    rel_L1_ones.append(l1_o);  L0_ones.append(l0_o)

rel_L1_std  = np.asarray(rel_L1_std)
rel_L1_var  = np.asarray(rel_L1_var)
rel_L1_ones = np.asarray(rel_L1_ones)

L0_std  = np.asarray(L0_std,  dtype=int)
L0_var  = np.asarray(L0_var,  dtype=int)
L0_ones = np.asarray(L0_ones, dtype=int)

print("\nSingle-trajectory ensemble results:")
print("  Mean relative L1 (no weighting):   ", np.mean(rel_L1_std))
print("  Mean relative L1 (variance GLS):   ", np.mean(rel_L1_var))
print("  Mean relative L1 (weights=1 GLS):  ", np.mean(rel_L1_ones))
print("  Mean L0 (no weighting):            ", np.mean(L0_std))
print("  Mean L0 (variance GLS):            ", np.mean(L0_var))
print("  Mean L0 (weights=1 GLS):           ", np.mean(L0_ones))

# %%
# Boxplot: relative L1 errors (single trajectory)
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data = [rel_L1_std, rel_L1_var, rel_L1_ones]
labels = ["No weighting", "Variance GLS", "Weights = 1 GLS"]
means = [np.mean(d) for d in data]

bp = ax.boxplot(
    data,
    labels=labels,
    showfliers=False,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)

colors = ["tab:blue", "tab:orange", "tab:green"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)

for x, m in zip([1, 2, 3], means):
    ax.scatter(x, m, marker="D", color="black", s=30, zorder=3)

ax.set_ylabel(r"Mean relative $L_1$ coefficient error")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()

# %%
# Boxplot: L0 support errors (single trajectory)
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_L0 = [L0_std, L0_var, L0_ones]
labels_L0 = ["No weighting", "Variance GLS", "Weights = 1 GLS"]
means_L0 = [np.mean(d) for d in data_L0]

bp = ax.boxplot(
    data_L0,
    labels=labels_L0,
    showfliers=False,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)

colors_L0 = ["tab:blue", "tab:orange", "tab:green"]
for patch, c in zip(bp["boxes"], colors_L0):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)

for x, m in zip([1, 2, 3], means_L0):
    ax.scatter(x, m, marker="D", color="black", s=30, zorder=3)

ax.set_ylabel(r"$L_0$ support error")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Monte Carlo Robustness Experiment

# Finally, we assess robustness across **multiple initial conditions**.
# For each run:

# 1. Sample a random initial condition
#    \[
#    y_0 = (\theta_0,\omega_0),\quad
#    \theta_0 \sim \mathcal U([-a,a]),\; \omega_0 \sim \mathcal U([-b,b]).
#    \]
# 2. Simulate a clean trajectory of length \(T_{\text{mc}} = 5\) s.
# 3. Build a heteroscedastic variance field from \(|\omega(t)|\) and
#    add noise with the same model as above.
# 4. Fit the three weak libraries with smaller ensembles and
#    **average coefficients over ensemble members**.
# 5. Compute per-run relative \(L_1\) and \(L_0\) errors vs the **true**
#    coefficients.

# We then summarize the error distributions across runs.

# %%
N_RUNS = 100
T_mc = 5.0
dt_mc = 1e-3

rng_mc = np.random.default_rng(999)

rel_L1_std_runs, rel_L1_var_runs, rel_L1_ones_runs = [], [], []
L0_std_runs, L0_var_runs, L0_ones_runs = [], [], []

for run in tqdm(range(N_RUNS), desc="Monte Carlo pendulum"):

    # 1) Random initial condition
    theta_bounds = np.array([-np.pi, np.pi])
    omega_bounds = np.array([-2.0, 2.0])

    theta0 = rng_mc.uniform(theta_bounds[0], theta_bounds[1])
    omega0 = rng_mc.uniform(omega_bounds[0], omega_bounds[1])
    y0_run = np.array([theta0, omega0])

    # 2) Clean trajectory
    t_run, Y_run_clean = simulate(y0_run, T=T_mc, dt=dt_mc)

    # 3) Heteroscedastic noise (same model)
    omega_mag_run = np.abs(Y_run_clean[:, 1])
    variance_run = (sigma0 + alpha_noise * omega_mag_run)**2
    variance_run = np.maximum(variance_run, 1e-10)
    std_run = np.sqrt(variance_run)

    noise_run = std_run[:, None] * rng_mc.standard_normal(size=Y_run_clean.shape)
    Y_run_noisy = Y_run_clean + noise_run

    # 4) Weak libraries for this run
    XT_run = t_run[:, None]
    base_lib_run = custom_library

    weak_lib_run = WeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=1,
        spatiotemporal_grid=XT_run,
        is_uniform=True,
        include_bias=False,
    )

    weighted_weak_lib_var_run = WeightedWeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=1,
        spatiotemporal_grid=XT_run,
        spatiotemporal_weights=variance_run,
        is_uniform=True,
        include_bias=False,
    )

    weighted_weak_lib_ones_run = WeightedWeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=1,
        spatiotemporal_grid=XT_run,
        spatiotemporal_weights=np.ones_like(variance_run),
        is_uniform=True,
        include_bias=False,
    )

    # 5) Fit SINDy models (smaller ensembles for speed)
    opt_std_run = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                       bagging=True, n_models=20)
    opt_var_run = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                       bagging=True, n_models=20)
    opt_ones_run = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                        bagging=True, n_models=20)

    model_std_run = ps.SINDy(feature_library=weak_lib_run,
                             optimizer=opt_std_run)
    model_var_run = ps.SINDy(feature_library=weighted_weak_lib_var_run,
                             optimizer=opt_var_run)
    model_ones_run = ps.SINDy(feature_library=weighted_weak_lib_ones_run,
                              optimizer=opt_ones_run)

    model_std_run.fit(Y_run_noisy, t=t_run)
    model_var_run.fit(Y_run_noisy, t=t_run)
    model_ones_run.fit(Y_run_noisy, t=t_run)

    # Ensemble-averaged coefficients for this run
    C_std_run  = np.mean(np.stack(opt_std_run.coef_list,  axis=0), axis=0).reshape(-1)
    C_var_run  = np.mean(np.stack(opt_var_run.coef_list,  axis=0), axis=0).reshape(-1)
    C_ones_run = np.mean(np.stack(opt_ones_run.coef_list, axis=0), axis=0).reshape(-1)

    l1_s, l0_s = coeff_errors(C_std_run,  C_true_flat)
    l1_v, l0_v = coeff_errors(C_var_run,  C_true_flat)
    l1_o, l0_o = coeff_errors(C_ones_run, C_true_flat)

    rel_L1_std_runs.append(l1_s);   L0_std_runs.append(l0_s)
    rel_L1_var_runs.append(l1_v);   L0_var_runs.append(l0_v)
    rel_L1_ones_runs.append(l1_o);  L0_ones_runs.append(l0_o)

rel_L1_std_runs  = np.asarray(rel_L1_std_runs)
rel_L1_var_runs  = np.asarray(rel_L1_var_runs)
rel_L1_ones_runs = np.asarray(rel_L1_ones_runs)

L0_std_runs  = np.asarray(L0_std_runs,  dtype=int)
L0_var_runs  = np.asarray(L0_var_runs,  dtype=int)
L0_ones_runs = np.asarray(L0_ones_runs, dtype=int)

print("\nMonte Carlo (per-run ensemble-averaged coefficients):")
print("  Mean relative L1 (no weighting):   ", np.mean(rel_L1_std_runs))
print("  Mean relative L1 (variance GLS):   ", np.mean(rel_L1_var_runs))
print("  Mean relative L1 (weights=1 GLS):  ", np.mean(rel_L1_ones_runs))
print("  Mean L0 (no weighting):            ", np.mean(L0_std_runs))
print("  Mean L0 (variance GLS):            ", np.mean(L0_var_runs))
print("  Mean L0 (weights=1 GLS):           ", np.mean(L0_ones_runs))

# %%
# Boxplot: Monte Carlo relative L1 errors
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_mc = [rel_L1_std_runs, rel_L1_var_runs, rel_L1_ones_runs]
labels_mc = ["No weighting", "Variance GLS", "Weights = 1 GLS"]
means_mc = [np.mean(d) for d in data_mc]

bp = ax.boxplot(
    data_mc,
    labels=labels_mc,
    showfliers=False,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)

colors_mc = ["tab:blue", "tab:orange", "tab:green"]
for patch, c in zip(bp["boxes"], colors_mc):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)

for x, m in zip([1, 2, 3], means_mc):
    ax.scatter(x, m, marker="D", color="black", s=30, zorder=3)

ax.set_ylabel(r"Mean relative $L_1$ coefficient error (per run)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()

# %%
# Boxplot: Monte Carlo L0 support errors
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_L0_mc = [L0_std_runs, L0_var_runs, L0_ones_runs]
labels_L0_mc = ["No weighting", "Variance GLS", "Weights = 1 GLS"]
means_L0_mc = [np.mean(d) for d in data_L0_mc]

bp = ax.boxplot(
    data_L0_mc,
    labels=labels_L0_mc,
    showfliers=False,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)

colors_L0_mc = ["tab:blue", "tab:orange", "tab:green"]
for patch, c in zip(bp["boxes"], colors_L0_mc):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)

for x, m in zip([1, 2, 3], means_L0_mc):
    ax.scatter(x, m, marker="D", color="black", s=30, zorder=3)

ax.set_ylabel(r"$L_0$ support error (per run)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()
