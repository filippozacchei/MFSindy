# %% [markdown]
# # Lorenz System with Heteroscedastic Noise and Weak SINDy + GLS
#
# We consider the **Lorenz system**
#
# \begin{aligned}
# \dot{x} &= \sigma (y - x), \\
# \dot{y} &= x(\rho - z) - y, \\
# \dot{z} &= x y - \beta z,
# \end{aligned}
#
# with parameters $\sigma, \rho, \beta > 0$.
#
# The system is simulated from an off–attractor initial condition and the
# resulting trajectory
#
# \[
# U(t) = [x(t), y(t), z(t)]^\top \in \mathbb{R}^3
# \]
#
# is used as the clean reference signal.
#
# We will:
#
# 1. Simulate a clean Lorenz trajectory.
# 2. Add heteroscedastic noise whose variance grows with the distance from the origin.
# 3. Build three weak SINDy libraries:
#    - no weighting,
#    - GLS weighting based on the variance field,
#    - GLS weighting with unit weights.
# 4. Compare coefficient errors (relative $L_1$ and support $L_0$) for a single trajectory.
# 5. Run a Monte Carlo robustness experiment over many random initial conditions.

# %%
# --- Imports and setup -------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm
from scipy.integrate import solve_ivp

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary, WeightedWeakPDELibrary  # custom GLS version

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})


# %% [markdown]
# ## 1. Lorenz System and True Coefficients

# %%
# --- Lorenz system definition ------------------------------------------
def lorenz(t, u, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = u
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


# --- True coefficients for degree-2 polynomial library -----------------
# Polynomial basis (degree 2, no bias) is:
# [x, y, z, x^2, x y, x z, y^2, y z, z^2]
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

true_coef = np.array([
    [-sigma,  sigma,  0.0,   0.0,  0.0,  0.0,  0.0,  0.0,   0.0],   # x'
    [ rho,   -1.0,   0.0,   0.0,  0.0, -1.0,  0.0,  0.0,   0.0],   # y'
    [ 0.0,    0.0,  -beta,  0.0,  1.0,  0.0,  0.0,  0.0,   0.0],   # z'
])


# %% [markdown]
# ## 2. Training Trajectory and Heteroscedastic Noise
#
# We simulate a single training trajectory from a random initial condition:
#
# \[
# U(t) = [x(t), y(t), z(t)]^\top.
# \]
#
# To emulate state-dependent measurement uncertainty, we introduce **heteroscedastic noise**:
#
# \[
# \mathrm{Var}[\varepsilon(t)] \propto \|U(t)\|_2,
# \]
#
# and construct noisy observations
#
# \[
# U_{\text{noisy}}(t) = U(t) + \varepsilon(t), \quad
# \varepsilon(t) \sim \mathcal{N}(0, \Sigma(t)),
# \]
#
# where $\Sigma(t)$ is diagonal with entries proportional to the local variance.

# %%
# --- Simulation parameters ---------------------------------------------
t0, t1 = 0.0, 10.0
dt = 1e-3
t_eval = np.arange(t0, t1, dt)

rng = np.random.default_rng(0)

# Random initial conditions for training and test trajectories
u0 = rng.uniform(-10.0, 10.0, size=3)
u0_test = rng.uniform(-10.0, 10.0, size=3)

print("Initial condition (train):", u0)
print("Initial condition (test) :", u0_test)

# --- Numerical integration (train and test) ----------------------------
sol = solve_ivp(
    lorenz, (t0, t1), u0, t_eval=t_eval,
    rtol=1e-12, atol=1e-12
)
U = sol.y.T   # shape (T, 3)

sol_test = solve_ivp(
    lorenz, (t0, t1), u0_test, t_eval=t_eval,
    rtol=1e-12, atol=1e-12
)
U_test = sol_test.y.T   # shape (T, 3)

# --- Heteroscedastic noise model ---------------------------------------
noise_level = 0.25

d = np.linalg.norm(U, axis=1)          # distance from origin
alpha = noise_level
eps = 1e-6                             # prevents zero variance

variance = (alpha * d)**2 + eps        # Var[ε(t)]
std = np.sqrt(variance)

noise = std[:, None] * rng.standard_normal(size=U.shape)
U_noisy = U + noise

print("Noise std range:", std.min(), "to", std.max())

# --- 3D plot: clean vs noisy trajectory --------------------------------
fig = plt.figure(figsize=(5, 4), dpi=150)
ax = fig.add_subplot(111, projection="3d")

ax.plot(U[:, 0], U[:, 1], U[:, 2],
        lw=1.2, color="black", alpha=0.9, label="clean")

ax.scatter(U_noisy[:, 0], U_noisy[:, 1], U_noisy[:, 2],
           s=3, color="steelblue", alpha=0.4, label="noisy")

ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
ax.set_box_aspect([1, 1, 1])

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    try:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    except AttributeError:
        axis.set_pane_color((1, 1, 1, 0))
    axis.line.set_color((0, 0, 0, 0))

ax.grid(False)
ax.set_facecolor((0, 0, 0, 0))

plt.tight_layout(pad=0)
plt.show()

# --- Time series vs noise level ----------------------------------------
fig, ax = plt.subplots(figsize=(6, 3), dpi=150)

ax.plot(t_eval, U_noisy[:, 0], label=r"$x(t)$ noisy", lw=1.0)
ax.plot(t_eval, std, label=r"$\sigma(t)$", lw=1.5)

for t in np.arange(0.0, t1 + 1e-12, 0.5):
    ax.axvline(t, color="gray", linewidth=0.4, alpha=0.7)

ax.set_xlabel("t [s]")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 3. Spatiotemporal Weighting in the Weak Form
#
# To incorporate the heteroscedastic uncertainty, we define **spatiotemporal
# variance weights** on the time grid:
#
# \[
# \sigma^2(t) = \alpha^2 \|U(t)\|_2^2 + \varepsilon.
# \]
#
# These weights define a whitening operator $W$ that rescales the weak
# integrals according to the local measurement uncertainty. The weighted weak
# SINDy regression solves
#
# \[
# \min_{\xi} \;\bigl\| W(\Theta \xi - V)\bigr\|_2^2,
# \]
#
# where $W$ is constructed from the variance field via a Cholesky factorization
# in the weak space, implementing a generalized least-squares fit.

# %%
# --- Spatiotemporal variance weights for weak library ------------------
spatiotemporal_weights = variance
spatiotemporal_weights_scaled = spatiotemporal_weights / np.mean(spatiotemporal_weights)

XT = t_eval[:, None]

base_library = ps.PolynomialLibrary(degree=2, include_bias=False)

weak_lib = WeakPDELibrary(
    function_library=base_library,
    derivative_order=1,
    spatiotemporal_grid=XT,
    is_uniform=True,
    H_xt=0.05,
    K=1000,
    include_bias=False,
)

weighted_weak_lib = WeightedWeakPDELibrary(
    function_library=base_library,
    derivative_order=1,
    spatiotemporal_grid=XT,
    spatiotemporal_weights=spatiotemporal_weights_scaled,
    is_uniform=True,
    K=1000,
    H_xt=0.05,
    include_bias=False,
)

weighted_weak_lib2 = WeightedWeakPDELibrary(
    function_library=base_library,
    derivative_order=1,
    spatiotemporal_grid=XT,
    spatiotemporal_weights=np.ones_like(spatiotemporal_weights),
    is_uniform=True,
    K=1000,
    H_xt=0.05,
    include_bias=False,
)

# --- Single-fit comparison on one trajectory ---------------------------
opt1 = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.5),
                            bagging=True, n_models=100)
opt2 = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.5),
                            bagging=True, n_models=100)
opt3 = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.5),
                            bagging=True, n_models=100)

model_weak      = ps.SINDy(feature_library=weak_lib,          optimizer=opt1).fit(U_noisy, t=t_eval)
model_weighted  = ps.SINDy(feature_library=weighted_weak_lib, optimizer=opt2).fit(U_noisy, t=t_eval)
model_weighted2 = ps.SINDy(feature_library=weighted_weak_lib2,optimizer=opt3).fit(U_noisy, t=t_eval)

print("\nWeak SINDy (no weighting):")
model_weak.print()
print("\nWeighted Weak SINDy (variance GLS):")
model_weighted.print()
print("\nWeighted Weak SINDy (weights = 1):")
model_weighted2.print()


# %% [markdown]
# ## 4. Ensemble Coefficients: Relative \(L_1\) and \(L_0\) Support Errors
#
# We now analyze the ensemble of coefficients from the single trajectory fit.
# For each ensemble member, we compute:
#
# - mean **relative** $L_1$ error on nonzero true coefficients,
# - $L_0$ support error (mismatch in sparsity pattern).

# %%
# --- Collect ensemble coefficient tensors ------------------------------
coef_weak      = np.stack(opt1.coef_list, axis=0)   # (E, 3, n_terms)
coef_weighted  = np.stack(opt2.coef_list, axis=0)
coef_weighted2 = np.stack(opt3.coef_list, axis=0)

print("coef_weak shape:     ", coef_weak.shape)
print("coef_weighted shape: ", coef_weighted.shape)
print("coef_weighted2 shape:", coef_weighted2.shape)


# --- Relative L1 and L0 support error helpers --------------------------
def coeff_errors(C_est, C_true, tol_support=1e-6, tol_rel=1e-8):
    """
    Returns mean relative L1 error on nonzero true coefficients + L0 support mismatch.

    Parameters
    ----------
    C_est, C_true : 1D arrays (flattened coefficient vectors)
    tol_support   : threshold for deciding nonzero support
    tol_rel       : floor to avoid division by very small true coefficients

    Returns
    -------
    l1_rel : float
        Mean relative L1 error over indices with |C_true_j| > tol_support:
        mean_j |C_est_j - C_true_j| / max(|C_true_j|, tol_rel)
    l0_err : int
        Number of indices where support differs between C_est and C_true
        (false positives + false negatives).
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


C_true_flat = true_coef.reshape(-1)

E, n_states, n_terms = coef_weak.shape

rel_L1_std,  rel_L1_wgt,  rel_L1_wgt2  = [], [], []
L0_std,      L0_wgt,      L0_wgt2      = [], [], []

for e in range(E):
    C_std_e  = coef_weak[e].reshape(-1)
    C_wgt_e  = coef_weighted[e].reshape(-1)
    C_wgt2_e = coef_weighted2[e].reshape(-1)

    l1_s, l0_s   = coeff_errors(C_std_e,  C_true_flat)
    l1_w, l0_w   = coeff_errors(C_wgt_e,  C_true_flat)
    l1_w2, l0_w2 = coeff_errors(C_wgt2_e, C_true_flat)

    rel_L1_std.append(l1_s);   L0_std.append(l0_s)
    rel_L1_wgt.append(l1_w);   L0_wgt.append(l0_w)
    rel_L1_wgt2.append(l1_w2); L0_wgt2.append(l0_w2)

rel_L1_std  = np.asarray(rel_L1_std)
rel_L1_wgt  = np.asarray(rel_L1_wgt)
rel_L1_wgt2 = np.asarray(rel_L1_wgt2)

L0_std  = np.asarray(L0_std,  dtype=int)
L0_wgt  = np.asarray(L0_wgt,  dtype=int)
L0_wgt2 = np.asarray(L0_wgt2, dtype=int)

print("Single-trajectory ensemble:")
print("  Mean relative L1 (no weighting):     ", np.mean(rel_L1_std))
print("  Mean relative L1 (variance GLS):     ", np.mean(rel_L1_wgt))
print("  Mean relative L1 (weights=1 GLS):    ", np.mean(rel_L1_wgt2))
print("  Mean L0 (no weighting):              ", np.mean(L0_std))
print("  Mean L0 (variance GLS):              ", np.mean(L0_wgt))
print("  Mean L0 (weights=1 GLS):             ", np.mean(L0_wgt2))

# --- Boxplot of relative L1 errors -------------------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data   = [rel_L1_std, rel_L1_wgt, rel_L1_wgt2]
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

# --- Boxplot of L0 support errors --------------------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_L0   = [L0_std, L0_wgt, L0_wgt2]
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
#
# We now repeat the identification over many random initial conditions:
#
# 1. Simulate a clean Lorenz trajectory.
# 2. Add heteroscedastic noise with the same variance model.
# 3. Fit the three weak libraries (smaller ensembles for speed).
# 4. Average coefficients over the ensemble for each method.
# 5. Compute per-run:
#    - mean relative $L_1$ coefficient error,
#    - $L_0$ support error.
#
# Finally, we compare the distributions of these metrics across runs.

# %%
# --- Monte Carlo setup -------------------------------------------------
N_RUNS = 200   # adjust as needed
t1_mc = 10.0
t_eval_mc = np.arange(t0, t1_mc, dt)

rng_mc = np.random.default_rng(123)

rel_L1_std_runs,  rel_L1_wgt_runs,  rel_L1_wgt2_runs  = [], [], []
L0_std_runs,      L0_wgt_runs,      L0_wgt2_runs      = [], [], []

C_true_flat = true_coef.reshape(-1)

for run in tqdm(range(N_RUNS), desc="Monte Carlo Lorenz"):
    # 1) Random initial condition
    u0_run = rng_mc.uniform(-10.0, 10.0, size=3)

    sol_run = solve_ivp(
        lorenz, (t0, t1_mc), u0_run, t_eval=t_eval_mc,
        rtol=1e-12, atol=1e-12
    )
    U_run = sol_run.y.T  # (T, 3)

    # 2) Heteroscedastic noise
    d_run = np.linalg.norm(U_run, axis=1)
    variance_run = (noise_level * d_run)**2
    variance_run = np.maximum(variance_run, 1e-8)
    std_run = np.sqrt(variance_run)

    noise_run = std_run[:, None] * rng_mc.standard_normal(size=U_run.shape)
    U_noisy_run = U_run + noise_run

    # 3) Weak libraries for this run
    XT_run = t_eval_mc[:, None]
    base_lib_run = ps.PolynomialLibrary(degree=2, include_bias=False)

    weak_lib_run = WeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=1,
        spatiotemporal_grid=XT_run,
        is_uniform=True,
        H_xt=0.05,
        K=1000,
        include_bias=False,
    )

    spatiotemporal_weights_scaled_run = variance_run / np.mean(variance_run)

    weighted_weak_lib_run = WeightedWeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=1,
        spatiotemporal_grid=XT_run,
        spatiotemporal_weights=spatiotemporal_weights_scaled_run,
        is_uniform=True,
        H_xt=0.05,
        K=1000,
        include_bias=False,
    )

    weighted_weak_lib2_run = WeightedWeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=1,
        spatiotemporal_grid=XT_run,
        spatiotemporal_weights=np.ones_like(variance_run),
        is_uniform=True,
        H_xt=0.05,
        K=1000,
        include_bias=False,
    )

    # 4) Fit SINDy models (smaller ensembles)
    opt1_mc = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.5),
                                   bagging=True, n_models=20)
    opt2_mc = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.5),
                                   bagging=True, n_models=20)
    opt3_mc = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.5),
                                   bagging=True, n_models=20)

    model_weak_mc      = ps.SINDy(feature_library=weak_lib_run,          optimizer=opt1_mc)
    model_weighted_mc  = ps.SINDy(feature_library=weighted_weak_lib_run, optimizer=opt2_mc)
    model_weighted2_mc = ps.SINDy(feature_library=weighted_weak_lib2_run,optimizer=opt3_mc)

    model_weak_mc.fit(U_noisy_run, t=t_eval_mc)
    model_weighted_mc.fit(U_noisy_run, t=t_eval_mc)
    model_weighted2_mc.fit(U_noisy_run, t=t_eval_mc)

    # 5) Average ensemble coefficients for this run and compute errors
    C_std_run   = np.mean(np.stack(opt1_mc.coef_list,   axis=0), axis=0).reshape(-1)
    C_wgt_run   = np.mean(np.stack(opt2_mc.coef_list,   axis=0), axis=0).reshape(-1)
    C_wgt2_run  = np.mean(np.stack(opt3_mc.coef_list,   axis=0), axis=0).reshape(-1)

    l1_s, l0_s   = coeff_errors(C_std_run,  C_true_flat)
    l1_w, l0_w   = coeff_errors(C_wgt_run,  C_true_flat)
    l1_w2, l0_w2 = coeff_errors(C_wgt2_run, C_true_flat)

    rel_L1_std_runs.append(l1_s);   L0_std_runs.append(l0_s)
    rel_L1_wgt_runs.append(l1_w);   L0_wgt_runs.append(l0_w)
    rel_L1_wgt2_runs.append(l1_w2); L0_wgt2_runs.append(l0_w2)

rel_L1_std_runs  = np.asarray(rel_L1_std_runs)
rel_L1_wgt_runs  = np.asarray(rel_L1_wgt_runs)
rel_L1_wgt2_runs = np.asarray(rel_L1_wgt2_runs)

L0_std_runs  = np.asarray(L0_std_runs,  dtype=int)
L0_wgt_runs  = np.asarray(L0_wgt_runs,  dtype=int)
L0_wgt2_runs = np.asarray(L0_wgt2_runs, dtype=int)

print("Monte Carlo (per-run ensemble-averaged coefficients):")
print("  Mean relative L1 (no weighting):     ", np.mean(rel_L1_std_runs))
print("  Mean relative L1 (variance GLS):     ", np.mean(rel_L1_wgt_runs))
print("  Mean relative L1 (weights=1 GLS):    ", np.mean(rel_L1_wgt2_runs))
print("  Mean L0 (no weighting):              ", np.mean(L0_std_runs))
print("  Mean L0 (variance GLS):              ", np.mean(L0_wgt_runs))
print("  Mean L0 (weights=1 GLS):             ", np.mean(L0_wgt2_runs))

# --- Boxplot of Monte Carlo relative L1 errors -------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_mc   = [rel_L1_std_runs, rel_L1_wgt_runs, rel_L1_wgt2_runs]
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

# --- Boxplot of Monte Carlo L0 support errors --------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

data_L0_mc   = [L0_std_runs, L0_wgt_runs, L0_wgt2_runs]
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
