# %% [markdown]
# # 1D Viscous Burgers Equation with Heteroscedastic Noise and Weak SINDy + GLS

# We consider the 1D viscous Burgers equation with periodic boundary conditions

# \begin{aligned}
# u_t + u\,u_x &= \nu\,u_{xx}, \qquad x \in [-L,L], \; t \ge 0,
# \end{aligned}

# with viscosity $\nu > 0$.

# We will:

# 1. Solve Burgers' equation numerically on a periodic spatial grid.
# 2. Add heteroscedastic noise whose variance is driven by the spatial gradient $|u_x|$.
# 3. Build three weak SINDy libraries:
#    - **No weighting** (standard weak SINDy),
#    - **Variance-based GLS weighting** (heteroscedastic weights),
#    - **Unit weights GLS** (normalization only via test functions).
# 4. On a single noisy simulation, compare ensemble coefficient errors
#    (relative $L_1$ and support $L_0$).
# 5. Run a Monte Carlo robustness experiment over random initial conditions and compare
#    the distributions of these errors across runs.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

from scipy.integrate import solve_ivp  # (used in other notebooks; kept for consistency)

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary, WeightedWeakPDELibrary

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# %% [markdown]
# ## 1. Burgers Solver (Finite Differences with Periodic BCs)

# We discretize space on a uniform periodic grid $x_j$ and time on $t^n$.
# Spatial derivatives are approximated via central differences:

# \[
# u_x \approx \frac{u_{j+1} - u_{j-1}}{2\Delta x}, \qquad
# u_{xx} \approx \frac{u_{j+1} - 2u_j + u_{j-1}}{\Delta x^2}.
# \]

# The semi-discrete update is

# \[
# u^{n+1}_j = u^n_j + \Delta t \left( -u^n_j u_x^n + \nu u_{xx}^n \right),
# \]

# with periodic wrap in space.

# %%
# ======================================================
# Burgers' PDE solver (periodic BCs, 1D in space)
# ======================================================
def burgers_solver(u0, tmax, dt, dx, nu=0.1):
    """
    Explicit finite-difference solver for 1D Burgers with periodic BCs.

    u_t + u u_x = nu u_xx
    """
    u = u0.copy()
    NT = int(tmax / dt)
    NX = len(u0)

    out = np.zeros((NT, NX))
    out[0] = u

    for n in range(1, NT):
        ux  = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        uxx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx * dx)
        u   = u + dt * (-u * ux + nu * uxx)
        out[n] = u

    return out


# %% [markdown]
# ## 2. Training Simulation and Heteroscedastic Noise
#
# We work on a periodic domain $x \in [-L, L]$ with $N_x$ points. The initial condition
# is a localized Gaussian bump. The clean solution $u(t,x)$ is computed on a grid
# of $N_t$ time steps.
#
# To mimic state-dependent measurement uncertainty, we use a variance field driven by
# the local gradient magnitude
#
# \[
# \sigma^2(t_i, x_j) = \bigl(\alpha\,|u_x(t_i,x_j)|\bigr)^2,
# \]
#
# and construct noisy observations
#
# \[
# u_{\text{noisy}}(t_i, x_j) = u(t_i, x_j)
# + \varepsilon(t_i,x_j), \qquad
# \varepsilon \sim \mathcal{N}(0, \sigma^2(t_i,x_j)).
# \]

# %%
# Spatial grid
L = 8.0
NX = 256
x = np.linspace(-L, L, NX, endpoint=False)
dx = x[1] - x[0]

# Time grid
dt = 1e-2
tmax = 10.0
time = np.arange(0.0, tmax, dt)
NT = len(time)

nu = 0.1  # viscosity used in the solver and in the true coefficients

# Initial condition: localized Gaussian bump
u0 = 0.5 * np.exp(-x**2)

# Clean solution u(t,x): shape (NT, NX)
U_clean = burgers_solver(u0, tmax, dt, dx, nu=nu)

# Reshape for PySINDy PDE: (Nt, Nx, n_states)
U = U_clean[:, :, None]  # (NT, NX, 1)

# Spatial gradient u_x for variance field
Ux = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2 * dx)  # (NT, NX, 1)
grad_mag = np.abs(Ux[:, :, 0])  # (NT, NX)

# Heteroscedastic variance driven by |u_x|
alpha = 0.25
variance = (alpha * grad_mag)**2  # (NT, NX)
std = np.sqrt(variance)

rng = np.random.default_rng(0)
noise = std[:, :, None] * rng.standard_normal(size=U.shape)
U_noisy = U + noise

print("U shape:        ", U.shape)
print("U_noisy shape:  ", U_noisy.shape)
print("variance shape: ", variance.shape)
print("std range:", std.min(), "to", std.max())

# %% [markdown]
# ### Clean vs Noisy Solution Surface

# %%
Xg, Tg = np.meshgrid(x, time)  # (NT, NX) each

fig = plt.figure(figsize=(6, 4), dpi=150)
ax = fig.add_subplot(111, projection="3d")

# Clean surface (white / gray)
white_surface = np.ones_like(U_clean.T)

ax.plot_surface(
    Xg, Tg, U_clean,
    facecolors=plt.cm.gray(white_surface),
    edgecolor="black",
    linewidth=0.15,
    antialiased=True,
    shade=False,
    alpha=1.0,
)

# Noisy surface (steelblue translucent)
ax.plot_surface(
    Xg, Tg, U_noisy[:, :, 0],
    color="steelblue",
    linewidth=0,
    antialiased=True,
    alpha=0.35,
)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_axis_off()

plt.tight_layout(pad=0)
plt.show()

# %% [markdown]
# ## 3. Weak SINDy Feature Libraries and True Burgers Coefficients

# Burgers' equation in the form

# \[
# u_t = -u u_x + \nu u_{xx}
# \]

# is identified using a polynomial PDE library of degree 2 (in $u$) and spatial
# derivatives up to order 2. For a single field $u$, the WeakPDELibrary generates
# features such as

# \[
# 1,\; u,\; u^2,\; u_x,\; u_{xx},\; u\,u_x,\; \dots
# \]

# The true coefficient vector is zero everywhere except at the convection term
# $u u_x$ (coefficient $-1$) and the diffusion term $u_{xx}$ (coefficient $\nu$).

# %%
# Weak convolution parameters in space-time
K = 1000    # number of weak test functions
# H_xt = 0.5  # (optional) width of test functions

# Spatiotemporal grid (Nt, Nx, 2)
X, T = np.meshgrid(x, time)            # (NT, NX)
XT = np.asarray([X, T]).transpose(1, 2, 0)  # (NT, NX, 2)

# Variance-based weights
spatiotemporal_weights = variance      # (NT, NX)
spatiotemporal_weights_scaled = spatiotemporal_weights / np.mean(spatiotemporal_weights)

base_library = ps.PolynomialLibrary(degree=2, include_bias=True)

weak_lib = WeakPDELibrary(
    function_library=base_library,
    derivative_order=2,
    spatiotemporal_grid=XT,
    is_uniform=True,
    K=K,
    include_bias=True,
)

weighted_weak_lib_var = WeightedWeakPDELibrary(
    function_library=base_library,
    derivative_order=2,
    spatiotemporal_grid=XT,
    spatiotemporal_weights=spatiotemporal_weights_scaled,
    is_uniform=True,
    K=K,
    include_bias=True,
)

weighted_weak_lib_ones = WeightedWeakPDELibrary(
    function_library=base_library,
    derivative_order=2,
    spatiotemporal_grid=XT,
    spatiotemporal_weights=np.ones_like(spatiotemporal_weights),
    is_uniform=True,
    K=K,
    include_bias=True,
)

# Ensemble optimizers
opt_std   = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                 n_models=100, bagging=True,
                                 n_subset=int(0.6 * NT))
opt_var   = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                 n_models=100, bagging=True,
                                 n_subset=int(0.6 * NT))
opt_ones  = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                 n_models=100, bagging=True,
                                 n_subset=int(0.6 * NT))

model_std   = ps.SINDy(feature_library=weak_lib,              optimizer=opt_std)
model_var   = ps.SINDy(feature_library=weighted_weak_lib_var, optimizer=opt_var)
model_ones  = ps.SINDy(feature_library=weighted_weak_lib_ones,optimizer=opt_ones)

model_std.fit(U_noisy, t=time)
model_var.fit(U_noisy, t=time)
model_ones.fit(U_noisy, t=time)

print("\n===== Weak SINDy (no weighting) =====")
model_std.print()
print("\n===== Weighted Weak SINDy (variance GLS) =====")
model_var.print()
print("\n===== Weighted Weak SINDy (weights = 1 GLS) =====")
model_ones.print()

feature_names = model_std.get_feature_names()
print("\nFeature names:")
for i, name in enumerate(feature_names):
    print(i, name)

# Identify indices of convection and diffusion terms in the feature list
# Adjust these if needed based on printed feature names.
try:
    # Common naming patterns in PySINDy PDE libraries
    if "u u_x" in feature_names:
        conv_idx = feature_names.index("u u_x")
    elif "u0 u0_x" in feature_names:
        conv_idx = feature_names.index("u0 u0_x")
    else:
        raise ValueError("Convection term not found; check feature_names.")

    if "u_xx" in feature_names:
        diff_idx = feature_names.index("u_xx")
    elif "u0_xx" in feature_names:
        diff_idx = feature_names.index("u0_xx")
    else:
        raise ValueError("Diffusion term not found; check feature_names.")

except ValueError as e:
    print("\n[Warning] Could not automatically locate convection/diffusion terms.")
    print("Please inspect `feature_names` above and set `conv_idx` and `diff_idx` manually.")
    raise e

print(f"\nConvection term index (u u_x): {conv_idx}")
print(f"Diffusion term index (u_xx):  {diff_idx}")

C_true = np.zeros(len(feature_names))
C_true[conv_idx] = -1.0
C_true[diff_idx] = nu

# %% [markdown]
# ## 4. Single-Trajectory Ensemble: Relative \(L_1\) and \(L_0\) Errors

# From each ensemble we obtain a set of coefficient vectors
# $\{\hat c^{(e)}\}_{e=1}^E$. For each ensemble member we compute:

# - mean **relative** $L_1$ error on nonzero true coefficients:

# \[
# \text{rel-}L_1
#   = \frac{1}{|\mathcal{I}|}
#     \sum_{j \in \mathcal{I}}
#     \frac{|\hat c_j - c_j^\star|}{\max(|c_j^\star|,\varepsilon)},
# \quad
# \mathcal{I} = \{ j : |c_j^\star| > \tau \},
# \]

# - $L_0$ support error

# \[
# L_0 = \#\{j : \mathbb{1}_{|\hat c_j|>\tau} \ne \mathbb{1}_{|c_j^\star|>\tau} \},
# \]

# counting false positives and false negatives relative to the true sparsity pattern.

# %%
# --- Stack ensemble coefficients: (E, 1, M) ----------------------------
coef_std   = np.stack(opt_std.coef_list,   axis=0)   # (E, 1, M)
coef_var   = np.stack(opt_var.coef_list,   axis=0)
coef_ones  = np.stack(opt_ones.coef_list,  axis=0)

print("coef_std shape:   ", coef_std.shape)
print("coef_var shape:   ", coef_var.shape)
print("coef_ones shape:  ", coef_ones.shape)

C_true_flat = C_true.reshape(-1)


def coeff_errors(C_est, C_true, tol_support=1e-6, tol_rel=1e-8):
    """
    Relative L1 error + L0 support mismatch for a single coefficient vector.

    C_est, C_true: 1D arrays.
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


E, _, M = coef_std.shape

rel_L1_std, rel_L1_var, rel_L1_ones = [], [], []
L0_std,     L0_var,     L0_ones     = [], [], []

for e in range(E):
    C_std_e   = coef_std[e].reshape(-1)
    C_var_e   = coef_var[e].reshape(-1)
    C_ones_e  = coef_ones[e].reshape(-1)

    l1_s, l0_s   = coeff_errors(C_std_e,  C_true_flat)
    l1_v, l0_v   = coeff_errors(C_var_e,  C_true_flat)
    l1_o, l0_o   = coeff_errors(C_ones_e, C_true_flat)

    rel_L1_std.append(l1_s);  L0_std.append(l0_s)
    rel_L1_var.append(l1_v);  L0_var.append(l0_v)
    rel_L1_ones.append(l1_o); L0_ones.append(l0_o)

rel_L1_std  = np.asarray(rel_L1_std)
rel_L1_var  = np.asarray(rel_L1_var)
rel_L1_ones = np.asarray(rel_L1_ones)

L0_std  = np.asarray(L0_std,  dtype=int)
L0_var  = np.asarray(L0_var,  dtype=int)
L0_ones = np.asarray(L0_ones, dtype=int)

print("Single-trajectory ensemble:")
print("  Mean relative L1 (no weighting):      ", np.mean(rel_L1_std))
print("  Mean relative L1 (variance GLS):      ", np.mean(rel_L1_var))
print("  Mean relative L1 (weights=1 GLS):     ", np.mean(rel_L1_ones))
print("  Mean L0 (no weighting):               ", np.mean(L0_std))
print("  Mean L0 (variance GLS):               ", np.mean(L0_var))
print("  Mean L0 (weights=1 GLS):              ", np.mean(L0_ones))

# --- Boxplot: relative L1 errors ---------------------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data   = [rel_L1_std, rel_L1_var, rel_L1_ones]
labels = ["No weighting", "Variance GLS", "Weights = 1 GLS"]
means  = [np.mean(d) for d in data]

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

# --- Boxplot: L0 support errors ----------------------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_L0   = [L0_std, L0_var, L0_ones]
labels_L0 = ["No weighting", "Variance GLS", "Weights = 1 GLS"]
means_L0  = [np.mean(d) for d in data_L0]

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
# ## 5. Monte Carlo Robustness Experiment

# To assess robustness, we repeat the identification over many random initial
# conditions. For each run:

# 1. Draw a random initial condition $u_0(x)$ (Gaussian bump with random
#    amplitude, center, and width).
# 2. Compute the clean Burgers solution $u(t,x)$.
# 3. Add heteroscedastic noise driven by $|u_x|$.
# 4. Fit the three weak libraries with a smaller ensemble for speed.
# 5. Average coefficients over the ensemble and compute per-run:
#    - mean relative $L_1$ coefficient error,
#    - $L_0$ support error.

# We then compare the distributions of these error metrics across runs.

# %%
def random_initial_condition(rng, x, L):
    """Random Gaussian bump IC for Monte Carlo experiments."""
    amp = rng.uniform(0.3, 1.0)
    center = rng.uniform(-L/2, L/2)
    width = rng.uniform(0.5, 1.5)
    return amp * np.exp(-(x - center)**2 / (2 * width**2))


N_RUNS = 100  # adjust as needed

rng_mc = np.random.default_rng(123)

rel_L1_std_runs,  rel_L1_var_runs,  rel_L1_ones_runs  = [], [], []
L0_std_runs,      L0_var_runs,      L0_ones_runs      = [], [], []

for run in tqdm(range(N_RUNS), desc="Monte Carlo Burgers"):
    # 1) Random initial condition
    u0_run = random_initial_condition(rng_mc, x, L)

    # 2) Clean solution
    U_run_clean = burgers_solver(u0_run, tmax, dt, dx, nu=nu)  # (NT, NX)
    U_run = U_run_clean[:, :, None]                            # (NT, NX, 1)

    # 3) Heteroscedastic noise based on |u_x|
    Ux_run = (np.roll(U_run, -1, axis=1) - np.roll(U_run, 1, axis=1)) / (2 * dx)
    grad_mag_run = np.abs(Ux_run[:, :, 0])  # (NT, NX)

    variance_run = (alpha * grad_mag_run)**2
    variance_run = np.maximum(variance_run, 1e-8)
    std_run = np.sqrt(variance_run)

    noise_run = std_run[:, :, None] * rng_mc.standard_normal(size=U_run.shape)
    U_noisy_run = U_run + noise_run

    # 4) Weak libraries for this run
    XT_run = XT  # same (time, x) grid

    base_lib_run = ps.PolynomialLibrary(degree=2, include_bias=True)

    weak_lib_run = WeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=2,
        spatiotemporal_grid=XT_run,
        is_uniform=True,
        K=500,
        include_bias=True,
    )

    weights_scaled_run = variance_run / np.mean(variance_run)

    weighted_weak_lib_var_run = WeightedWeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=2,
        spatiotemporal_grid=XT_run,
        spatiotemporal_weights=weights_scaled_run,
        is_uniform=True,
        K=500,
        include_bias=True,
    )

    weighted_weak_lib_ones_run = WeightedWeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=2,
        spatiotemporal_grid=XT_run,
        spatiotemporal_weights=np.ones_like(variance_run),
        is_uniform=True,
        K=500,
        include_bias=True,
    )

    # 5) Fit SINDy models (smaller ensembles for speed)
    opt_std_run  = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                        n_models=20, bagging=True,
                                        n_subset=int(0.6 * NT))
    opt_var_run  = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                        n_models=20, bagging=True,
                                        n_subset=int(0.6 * NT))
    opt_ones_run = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.05),
                                        n_models=20, bagging=True,
                                        n_subset=int(0.6 * NT))

    model_std_run  = ps.SINDy(feature_library=weak_lib_run,              optimizer=opt_std_run)
    model_var_run  = ps.SINDy(feature_library=weighted_weak_lib_var_run, optimizer=opt_var_run)
    model_ones_run = ps.SINDy(feature_library=weighted_weak_lib_ones_run,optimizer=opt_ones_run)

    model_std_run.fit(U_noisy_run, t=time)
    model_var_run.fit(U_noisy_run, t=time)
    model_ones_run.fit(U_noisy_run, t=time)

    # 6) Average ensemble coefficients and compute errors
    C_std_run  = np.mean(np.stack(opt_std_run.coef_list,  axis=0), axis=0).reshape(-1)
    C_var_run  = np.mean(np.stack(opt_var_run.coef_list,  axis=0), axis=0).reshape(-1)
    C_ones_run = np.mean(np.stack(opt_ones_run.coef_list, axis=0), axis=0).reshape(-1)

    l1_s, l0_s   = coeff_errors(C_std_run,  C_true_flat)
    l1_v, l0_v   = coeff_errors(C_var_run,  C_true_flat)
    l1_o, l0_o   = coeff_errors(C_ones_run, C_true_flat)

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
print("  Mean relative L1 (no weighting):     ", np.mean(rel_L1_std_runs))
print("  Mean relative L1 (variance GLS):     ", np.mean(rel_L1_var_runs))
print("  Mean relative L1 (weights=1 GLS):    ", np.mean(rel_L1_ones_runs))
print("  Mean L0 (no weighting):              ", np.mean(L0_std_runs))
print("  Mean L0 (variance GLS):              ", np.mean(L0_var_runs))
print("  Mean L0 (weights=1 GLS):             ", np.mean(L0_ones_runs))

# --- Boxplot: Monte Carlo relative L1 errors ---------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_mc   = [rel_L1_std_runs, rel_L1_var_runs, rel_L1_ones_runs]
labels_mc = ["No weighting", "Variance GLS", "Weights = 1 GLS"]
means_mc  = [np.mean(d) for d in data_mc]

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

# --- Boxplot: Monte Carlo L0 support errors ----------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_L0_mc   = [L0_std_runs, L0_var_runs, L0_ones_runs]
labels_L0_mc = ["No weighting", "Variance GLS", "Weights = 1 GLS"]
means_L0_mc  = [np.mean(d) for d in data_L0_mc]

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
