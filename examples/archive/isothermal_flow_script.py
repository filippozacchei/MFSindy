# %% [markdown]
# # 2D Isothermal Compressible Flow — Weak SINDy with GLS Weighting
#
# We consider the 2D **isothermal compressible Navier–Stokes** system
# (non-dimensionalized) for velocity $(u,v)$ and density $\rho$:
#
# \begin{aligned}
# u_t &= -(u u_x + v u_y) - \frac{1}{\rho}\left(p_x - \mu(\,u_{xx} + u_{yy}\,)\right),\\[4pt]
# v_t &= -(u v_x + v v_y) - \frac{1}{\rho}\left(p_y - \mu(\,v_{xx} + v_{yy}\,)\right),\\[4pt]
# \rho_t &= -\left(\frac{u p_x + v p_y}{RT} + \rho u_x + \rho v_y\right),
# \end{aligned}
#
# with isothermal pressure $p = \rho\,RT$, viscosity $\mu$, and gas constant–temperature
# product $RT$.
#
# We will:
#
# 1. Simulate the system from a **Taylor–Green–like vortex** initial condition.
# 2. Add **heteroscedastic measurement noise** whose variance grows with the local
#    velocity gradient magnitude.
# 3. Construct three weak SINDy libraries:
#    - **No weighting** (standard weak SINDy),
#    - **Variance-based GLS weighting** (heteroscedastic weights),
#    - **Unit weights GLS** (weights = 1 on all weak constraints).
# 4. Use a **clean reference trajectory** to obtain a “ground truth” coefficient
#    vector in a polynomial PDE library.
# 5. On noisy data, compare coefficient recovery via
#    - mean relative \(L_1\) coefficient error,
#    - \(L_0\) support error.
# 6. Run a **Monte Carlo experiment** over random perturbations of the initial
#    condition and report error distributions.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

import pysindy as ps
from pysindy.feature_library import WeakPDELibrary, WeightedWeakPDELibrary

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

# %% [markdown]
# ## 1. Isothermal Compressible Flow Generator
#
# We use a finite-difference time integrator for the 2D isothermal compressible
# Navier–Stokes equations on a periodic square domain $[0,L]\times[0,L]$ with a
# uniform grid of size $N\times N$.
#
# The helper `generate_compressible_flow` returns trajectories of
# \[
# U(t,x,y) = [u(t,x,y), v(t,x,y), \rho(t,x,y)]^\top
# \]
# on the spatiotemporal grid.

# %%
# ---------------------------------------------------------------------
# Compressible isothermal Navier–Stokes in 2D (RHS)
# ---------------------------------------------------------------------
def compressible(t, U, dx, N, mu, RT):
    """
    U is flattened (3 * N * N,),
    we reshape to (N, N, 3) with components (u, v, rho).
    """
    uvr = U.reshape(N, N, 3)
    u = uvr[:, :, 0]
    v = uvr[:, :, 1]
    rho = uvr[:, :, 2]

    FD1x = ps.differentiation.FiniteDifference(d=1, axis=0, periodic=True)
    FD1y = ps.differentiation.FiniteDifference(d=1, axis=1, periodic=True)
    FD2x = ps.differentiation.FiniteDifference(d=2, axis=0, periodic=True)
    FD2y = ps.differentiation.FiniteDifference(d=2, axis=1, periodic=True)

    ux  = FD1x._differentiate(u, dx)
    uy  = FD1y._differentiate(u, dx)
    uxx = FD2x._differentiate(u, dx)
    uyy = FD2y._differentiate(u, dx)

    vx  = FD1x._differentiate(v, dx)
    vy  = FD1y._differentiate(v, dx)
    vxx = FD2x._differentiate(v, dx)
    vyy = FD2y._differentiate(v, dx)

    p   = rho * RT
    px  = FD1x._differentiate(p, dx)
    py  = FD1y._differentiate(p, dx)

    ret = np.zeros_like(uvr)
    # u_t
    ret[:, :, 0] = -(u * ux + v * uy) - (px - mu * (uxx + uyy)) / rho
    # v_t
    ret[:, :, 1] = -(u * vx + v * vy) - (py - mu * (vxx + vyy)) / rho
    # rho_t
    ret[:, :, 2] = -(u * px / RT + v * py / RT + rho * ux + rho * vy)

    return ret.reshape(-1)


# ---------------------------------------------------------------------
# Initial condition generator
# ---------------------------------------------------------------------
def make_initial_condition(X, Y, L, ic_type="taylor-green", perturb_scale=1.0, rng=None):
    """
    Return (U0, V0, RHO0) for a chosen flow configuration.

    X, Y : meshgrid on [0,L]x[0,L]
    """
    if rng is None:
        rng = np.random.default_rng()

    if ic_type == "taylor-green":
        U0 = (-np.sin(2 * np.pi / L * X) + 0.5 * np.cos(2 * 2 * np.pi / L * Y))
        V0 = (0.5 * np.cos(2 * np.pi / L * X) - np.sin(2 * 2 * np.pi / L * Y))
        RHO0 = 1.0 + 0.5 * np.cos(2 * np.pi / L * X) * np.cos(2 * 2 * np.pi / L * Y)

        perturb = perturb_scale * np.exp(
            -((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (0.1 * L) ** 2
        ) * (0.5 * rng.standard_normal(U0.shape))

        U0 += perturb
        V0 -= perturb

    elif ic_type == "shear-layer":
        U0 = np.tanh((Y - L / 2) / 0.1)
        V0 = 0.05 * np.sin(2 * np.pi * X / L)
        RHO0 = 1.0 + 0.1 * np.exp(-((Y - L / 2) ** 2) / (0.1**2))

    else:
        raise ValueError(f"Unknown initial condition: {ic_type}")

    return U0, V0, RHO0


# ---------------------------------------------------------------------
# Flow field generator (single or multiple trajectories)
# ---------------------------------------------------------------------
def generate_compressible_flow(
    n_traj=1,
    N=64,
    Nt=101,
    L=5.0,
    T=2.0,
    mu=1.0,
    RT=1.0,
    noise_level=0.0,
    seed=42,
    initial_condition="taylor-green",
    noise_ic=0.1,
):
    """
    Generate one or more trajectories for isothermal 2D compressible flow.

    Returns
    -------
    trajectories : list of ndarray
        Each element has shape (N, N, Nt, 3) with components (u, v, rho).
    grid : ndarray of shape (N, N, Nt, 3)
        Spatiotemporal grid with entries (x, y, t).
    ts : list of ndarray
        Time arrays, each of length Nt.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, Nt)
    x = np.linspace(0.0, L, N, endpoint=False)
    y = np.linspace(0.0, L, N, endpoint=False)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    trajectories = []
    ts = []

    for _ in range(n_traj):
        U0, V0, RHO0 = make_initial_condition(X, Y, L, ic_type=initial_condition, rng=rng)
        noise_ic_arr = noise_ic * rng.standard_normal((N, N, 3))

        y0 = np.stack([U0, V0, RHO0], axis=-1) + noise_ic_arr

        sol = solve_ivp(
            compressible,
            (t[0], t[-1]),
            y0=y0.reshape(-1),
            t_eval=t,
            args=(dx, N, mu, RT),
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )

        u_field = sol.y.reshape(N, N, 3, -1).transpose(0, 1, 3, 2)  # (N, N, Nt, 3)

        if noise_level > 0.0:
            u_field += noise_level * rng.standard_normal(size=u_field.shape)

        trajectories.append(u_field)
        ts.append(t)

    # Spatiotemporal grid (same for all trajectories)
    grid = np.zeros((N, N, Nt, 3))
    grid[:, :, :, 0] = X[:, :, None]
    grid[:, :, :, 1] = Y[:, :, None]
    grid[:, :, :, 2] = t[None, None, :]

    return trajectories, grid, ts


# %% [markdown]
# ## 2. Reference Simulation (Clean) and Heteroscedastic Noise
#
# We simulate a single trajectory from a Taylor–Green–like initial condition
# and treat it as the clean reference solution.
#
# To emulate state-dependent measurement uncertainty, we add **heteroscedastic
# noise** with variance proportional to the local velocity gradient magnitude:
#
# \[
# \sigma^2(t,x,y) = \bigl(\sigma_0 + \alpha \|\nabla u\|_2\bigr)^2,
# \]
#
# where $\nabla u$ collects the spatial derivatives of the velocity components
# $(u,v)$.
#
# Noisy observations are then
# \[
# U_{\text{noisy}}(t, x, y) = U(t, x, y) + \varepsilon(t, x, y),
# \qquad
# \varepsilon \sim \mathcal N(0,\,\sigma^2 I_3).
# \]

# %%
# Reference simulation (no measurement noise)
L = 5.0
N = 48          # spatial resolution
Nt = 51         # number of time snapshots
T = 1.0
mu = 1.0
RT = 1.0

trajectories, grid, ts = generate_compressible_flow(
    n_traj=1,
    N=N,
    Nt=Nt,
    L=L,
    T=T,
    mu=mu,
    RT=RT,
    noise_level=0.0,           # clean reference
    seed=1,
    initial_condition="taylor-green",
    noise_ic=0.05,
)

u_field = trajectories[0]  # (N, N, Nt, 3)
t = ts[0]                  # (Nt,)

print("u_field shape (N, N, Nt, 3):", u_field.shape)
print("t shape:", t.shape)
print("grid shape:", grid.shape)

# Rearrange to (Nt, Nx, Ny, n_states) for SINDy
U_clean = np.moveaxis(u_field, 2, 0)  # (Nt, N, N, 3)
XT = np.moveaxis(grid, 2, 0)          # (Nt, N, N, 3) with coords (x,y,t)

# %%
# --- Heteroscedastic noise field based on velocity gradients ----------
# Extract velocity components
u = U_clean[..., 0]  # (Nt, N, N)
v = U_clean[..., 1]

# Spatial step (uniform in x and y)
x_coords = grid[:, 0, 0, 0]
dx = x_coords[1] - x_coords[0]
dy = dx

# Central differences with periodic BCs
def ddx(f, h):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * h)

def ddy(f, h):
    return (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2 * h)

ux = ddx(u, dx)
uy = ddy(u, dy)
vx = ddx(v, dx)
vy = ddy(v, dy)

grad_mag = np.sqrt(ux**2 + uy**2 + vx**2 + vy**2)  # (Nt, N, N)

sigma0 = 1e-3
alpha = 0.5   # controls noise level

variance = (sigma0 + alpha * grad_mag)**2          # (Nt, N, N)
std = np.sqrt(variance)

rng = np.random.default_rng(123)
noise = std[..., None] * rng.standard_normal(size=U_clean.shape)
U_noisy = U_clean + noise

print("variance shape:", variance.shape)
print("std range:", std.min(), "to", std.max())
print("U_noisy shape:", U_noisy.shape)

# %%
# --- Visualize a snapshot: |u| and noise std --------------------------
snap = Nt // 2

vel_mag = np.sqrt(
    U_clean[snap, :, :, 0] ** 2 + U_clean[snap, :, :, 1] ** 2
)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=150)

im0 = axes[0].imshow(
    vel_mag,
    origin="lower",
    extent=(0, L, 0, L),
    cmap="viridis",
)
axes[0].set_title(r"$|\mathbf{u}(x,y,t)|$ (clean)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(
    std[snap],
    origin="lower",
    extent=(0, L, 0, L),
    cmap="magma",
)
axes[1].set_title(r"Noise std $\sigma(x,y,t)$")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Spatiotemporal Weighting in the Weak PDE Form
#
# Let $\Theta(U)$ be the weak PDE library evaluated on the spatiotemporal grid
# and $V(U_t)$ the weak time derivative integrals. In the presence of
# heteroscedastic measurement noise with variance field $\sigma^2(x,y,t)$, a
# natural weak generalized least-squares problem is
#
# \[
# \min_{\Xi} \; \bigl\| W(\Theta \Xi - V) \bigr\|_2^2,
# \]
#
# where the whitener $W$ is derived from the covariance of the weak residuals.
# In practice, in `WeightedWeakPDELibrary`, we supply a **variance field**
#
# \[
# \sigma^2(x,y,t) \propto \|\nabla u(x,y,t)\|_2,
# \]
#
# defined on the same spatiotemporal grid as the state, and the library
# constructs a Cholesky-based GLS weighting internally.
#
# We compare three settings:
#
# 1. **No weighting** — standard weak SINDy.
# 2. **Variance GLS** — GLS weights from the heteroscedastic variance field.
# 3. **Unit GLS** — weights identically one (whitening only via test functions).

# %% [markdown]
# ## 4. Reference “Ground Truth” Coefficients from Clean Data
#
# Because the analytic PDE in the chosen polynomial basis involves rational
# expressions in $\rho$, we do **not** hand-code the exact coefficient tensor
# as in the Lorenz or Burgers cases. Instead, we:
#
# 1. Fit a **reference weak PDE model** on the *clean* trajectory using
#    standard weak SINDy.
# 2. Treat the resulting coefficient tensor as the “ground truth”
#    $\Xi^\star$ for subsequent noisy experiments.
#
# This is a standard surrogate approach when the exact PDE in the chosen
# function library is algebraically complicated.

# %%
# --- Build reference weak library and model on clean data --------------
base_library = ps.PolynomialLibrary(degree=2, include_bias=True)

weak_lib_ref = WeakPDELibrary(
    function_library=base_library,
    derivative_order=2,
    spatiotemporal_grid=XT,   # (Nt, N, N, 3) with coords (x,y,t)
    is_uniform=True,
    K=500,
    include_bias=True,
)

opt_ref = ps.STLSQ(threshold=0.1)
model_ref = ps.SINDy(feature_library=weak_lib_ref, optimizer=opt_ref)

model_ref.fit(U_clean, t=t)

print("\nReference model (clean data):")
model_ref.print()

true_coef = model_ref.optimizer.coef_.copy()  # shape (n_states=3, n_terms)
C_true_flat = true_coef.reshape(-1)
print("\ntrue_coef shape:", true_coef.shape)

# %% [markdown]
# ## 5. Single-Trajectory Ensemble on Noisy Data
#
# We now fit **three ensemble weak models** on the noisy trajectory:
#
# - **No weighting**: standard weak SINDy.
# - **Variance GLS**: heteroscedastic GLS with variance field from $\|\nabla u\|$.
# - **Unit GLS**: GLS with constant weights equal to one.
#
# For each ensemble member we obtain a coefficient tensor
# $\hat\Xi^{(e)} \in \mathbb{R}^{3 \times n_{\text{terms}}}$.
# Comparing to $\Xi^\star$ we compute:
#
# - Mean **relative** \(L_1\) error over nonzero entries of $\Xi^\star$.
# - \(L_0\) support error (false positives + false negatives).

# %%
# --- Weak and weighted libraries for noisy data ------------------------
weak_lib = WeakPDELibrary(
    function_library=base_library,
    derivative_order=2,
    spatiotemporal_grid=XT,
    is_uniform=True,
    K=500,
    include_bias=True,
)

weighted_weak_lib_var = WeightedWeakPDELibrary(
    function_library=base_library,
    derivative_order=2,
    spatiotemporal_grid=XT,
    spatiotemporal_weights=variance,   # (Nt, N, N)
    is_uniform=True,
    K=500,
    include_bias=True,
)

weighted_weak_lib_ones = WeightedWeakPDELibrary(
    function_library=base_library,
    derivative_order=2,
    spatiotemporal_grid=XT,
    spatiotemporal_weights=np.ones_like(variance),
    is_uniform=True,
    K=500,
    include_bias=True,
)

# Ensemble optimizers
opt_std  = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.1),
                                n_models=40, bagging=True,
                                n_subset=int(0.6 * Nt))
opt_var  = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.1),
                                n_models=40, bagging=True,
                                n_subset=int(0.6 * Nt))
opt_ones = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.1),
                                n_models=40, bagging=True,
                                n_subset=int(0.6 * Nt))

model_std  = ps.SINDy(feature_library=weak_lib,              optimizer=opt_std)
model_var  = ps.SINDy(feature_library=weighted_weak_lib_var, optimizer=opt_var)
model_ones = ps.SINDy(feature_library=weighted_weak_lib_ones,optimizer=opt_ones)

model_std.fit(U_noisy, t=t)
model_var.fit(U_noisy, t=t)
model_ones.fit(U_noisy, t=t)

print("\n===== Weak SINDy (no weighting) =====")
model_std.print()
print("\n===== Weighted Weak SINDy (variance GLS) =====")
model_var.print()
print("\n===== Weighted Weak SINDy (weights = 1 GLS) =====")
model_ones.print()

# --- Stack ensemble coefficients: (E, n_states, n_terms) ---------------
coef_std   = np.stack(opt_std.coef_list,   axis=0)
coef_var   = np.stack(opt_var.coef_list,   axis=0)
coef_ones  = np.stack(opt_ones.coef_list,  axis=0)

print("coef_std shape:  ", coef_std.shape)
print("coef_var shape:  ", coef_var.shape)
print("coef_ones shape: ", coef_ones.shape)

# %%
def coeff_errors(C_est, C_true, tol_support=1e-6, tol_rel=1e-8):
    """
    Relative L1 error + L0 support mismatch for a single coefficient vector.

    C_est, C_true : 1D arrays of same length.
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


E, n_states, n_terms = coef_std.shape

rel_L1_std,  rel_L1_var,  rel_L1_ones  = [], [], []
L0_std,      L0_var,      L0_ones      = [], [], []

for e in range(E):
    C_std_e  = coef_std[e].reshape(-1)
    C_var_e  = coef_var[e].reshape(-1)
    C_ones_e = coef_ones[e].reshape(-1)

    l1_s, l0_s   = coeff_errors(C_std_e,  C_true_flat)
    l1_v, l0_v   = coeff_errors(C_var_e,  C_true_flat)
    l1_o, l0_o   = coeff_errors(C_ones_e, C_true_flat)

    rel_L1_std.append(l1_s);   L0_std.append(l0_s)
    rel_L1_var.append(l1_v);   L0_var.append(l0_v)
    rel_L1_ones.append(l1_o);  L0_ones.append(l0_o)

rel_L1_std  = np.asarray(rel_L1_std)
rel_L1_var  = np.asarray(rel_L1_var)
rel_L1_ones = np.asarray(rel_L1_ones)

L0_std  = np.asarray(L0_std,  dtype=int)
L0_var  = np.asarray(L0_var,  dtype=int)
L0_ones = np.asarray(L0_ones, dtype=int)

print("\nSingle-trajectory ensemble (noisy data):")
print("  Mean relative L1 (no weighting):      ", np.mean(rel_L1_std))
print("  Mean relative L1 (variance GLS):      ", np.mean(rel_L1_var))
print("  Mean relative L1 (weights=1 GLS):     ", np.mean(rel_L1_ones))
print("  Mean L0 (no weighting):               ", np.mean(L0_std))
print("  Mean L0 (variance GLS):               ", np.mean(L0_var))
print("  Mean L0 (weights=1 GLS):              ", np.mean(L0_ones))

# %%
# --- Boxplot: relative L1 errors (single trajectory) -------------------
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

# %%
# --- Boxplot: L0 support errors (single trajectory) --------------------
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
# ## 6. Monte Carlo Robustness Experiment
#
# We now repeat the identification over multiple runs with **different
# initial perturbations**. For each run:
#
# 1. Generate a new trajectory by re-calling the compressible solver with random
#    perturbations of the Taylor–Green initial condition.
# 2. Construct a heteroscedastic variance field from the velocity gradients.
# 3. Fit the three weak SINDy libraries (smaller ensembles for speed).
# 4. Average coefficients over the ensemble in each method and compute:
#    - mean relative \(L_1\) error vs the reference coefficients,
#    - \(L_0\) support error vs the reference support.
#
# We then compare the distributions of these errors across runs.

# %%
N_RUNS = 30    # adjust as needed
alpha_mc = alpha   # reuse same noise slope

rel_L1_std_runs,  rel_L1_var_runs,  rel_L1_ones_runs  = [], [], []
L0_std_runs,      L0_var_runs,      L0_ones_runs      = [], [], []

rng_mc = np.random.default_rng(999)

for run in tqdm(range(N_RUNS), desc="Monte Carlo compressible flow"):

    # --- New trajectory (clean) ----------------------------------------
    seed_run = int(rng_mc.integers(1, 10_000))
    trajectories_run, grid_run, ts_run = generate_compressible_flow(
        n_traj=1,
        N=N,
        Nt=Nt,
        L=L,
        T=T,
        mu=mu,
        RT=RT,
        noise_level=0.0,
        seed=seed_run,
        initial_condition="taylor-green",
        noise_ic=0.05,
    )

    u_field_run = trajectories_run[0]          # (N, N, Nt, 3)
    t_run = ts_run[0]                          # (Nt,)
    U_run_clean = np.moveaxis(u_field_run, 2, 0)  # (Nt, N, N, 3)
    XT_run = np.moveaxis(grid_run, 2, 0)          # (Nt, N, N, 3)

    # --- Heteroscedastic noise for this run ----------------------------
    u_run = U_run_clean[..., 0]
    v_run = U_run_clean[..., 1]

    x_coords_run = grid_run[:, 0, 0, 0]
    dx_run = x_coords_run[1] - x_coords_run[0]
    dy_run = dx_run

    ux_run = ddx(u_run, dx_run)
    uy_run = ddy(u_run, dy_run)
    vx_run = ddx(v_run, dx_run)
    vy_run = ddy(v_run, dy_run)

    grad_mag_run = np.sqrt(ux_run**2 + uy_run**2 + vx_run**2 + vy_run**2)
    variance_run = (sigma0 + alpha_mc * grad_mag_run)**2
    variance_run = np.maximum(variance_run, 1e-10)
    std_run = np.sqrt(variance_run)

    noise_run = std_run[..., None] * rng_mc.standard_normal(size=U_run_clean.shape)
    U_run_noisy = U_run_clean + noise_run

    # --- Weak libraries for this run -----------------------------------
    base_lib_run = ps.PolynomialLibrary(degree=2, include_bias=True)

    weak_lib_run = WeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=2,
        spatiotemporal_grid=XT_run,
        is_uniform=True,
        K=400,
        include_bias=True,
    )

    weighted_weak_lib_var_run = WeightedWeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=2,
        spatiotemporal_grid=XT_run,
        spatiotemporal_weights=variance_run,
        is_uniform=True,
        K=400,
        include_bias=True,
    )

    weighted_weak_lib_ones_run = WeightedWeakPDELibrary(
        function_library=base_lib_run,
        derivative_order=2,
        spatiotemporal_grid=XT_run,
        spatiotemporal_weights=np.ones_like(variance_run),
        is_uniform=True,
        K=400,
        include_bias=True,
    )

    # --- Fit SINDy models (smaller ensembles) --------------------------
    opt_std_run  = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.1),
                                        n_models=15, bagging=True,
                                        n_subset=int(0.6 * Nt))
    opt_var_run  = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.1),
                                        n_models=15, bagging=True,
                                        n_subset=int(0.6 * Nt))
    opt_ones_run = ps.EnsembleOptimizer(ps.STLSQ(threshold=0.1),
                                        n_models=15, bagging=True,
                                        n_subset=int(0.6 * Nt))

    model_std_run  = ps.SINDy(feature_library=weak_lib_run,              optimizer=opt_std_run)
    model_var_run  = ps.SINDy(feature_library=weighted_weak_lib_var_run, optimizer=opt_var_run)
    model_ones_run = ps.SINDy(feature_library=weighted_weak_lib_ones_run,optimizer=opt_ones_run)

    model_std_run.fit(U_run_noisy, t=t_run)
    model_var_run.fit(U_run_noisy, t=t_run)
    model_ones_run.fit(U_run_noisy, t=t_run)

    # --- Ensemble-averaged coefficients for this run -------------------
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

# %%
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

# %%
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
