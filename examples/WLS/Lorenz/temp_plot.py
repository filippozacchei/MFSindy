import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle

from generator import generate_lorenz_data
import numpy as np
import pysindy as ps
# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.linewidth": 0.9
})

# Generate ONE high-fidelity reference trajectory
X_hist_list, _, t_hist_list = generate_lorenz_data(
    n_traj=1,
    noise_level=0.01,
    T=1.0
)

# Convert lists → arrays
X_hist = X_hist_list[0]            # shape (T, 3)
t_hist = np.array(t_hist_list[0])  # shape (T,)

colors = ["#4e79a7", "#a17d4f", "#e15759"]  # x,y,z
c_u = (0.55,0.55,0.55)       # MF (unweighted)
from matplotlib.cm import plasma
c_w = plasma(0.6)           # MF-W (weighted)
# Save
np.save("X_hist.npy", X_hist)
np.save("t_hist.npy", t_hist)

print("✅ Saved X_hist.npy and t_hist.npy")
print("X_hist shape:", X_hist.shape)
print("t_hist shape:", t_hist.shape)

library = ps.PolynomialLibrary(degree=2)

# ------------------------------------------------------------
# LOAD REAL COEFFICIENT SAMPLES
# ------------------------------------------------------------
C_mf  = np.load("C_mf_samples.npy")   # (N, n_terms, 3)
C_mfw = np.load("C_mfw_samples.npy")  # (N, n_terms, 3)

# Load history used to initialize forecast
X_hist = np.load("X_hist.npy")        # (T, 3)
t_hist = np.load("t_hist.npy")

dt = t_hist[1] - t_hist[0]
T_pred = 1.0
steps_pred = int(T_pred / dt)
t_pred = np.linspace(0, T_pred, steps_pred)

u0 = X_hist[-1]     # initial condition (3,)
print(u0)

# ------------------------------------------------------------
# HISTOGRAMS OF TOP-4 MOST IMPORTANT COEFFICIENTS
# ------------------------------------------------------------

# Flatten coefficients: (N, n_terms*3)
mf_flat  = C_mf.reshape(C_mf.shape[0],  - 1)
mfw_flat = C_mfw.reshape(C_mfw.shape[0], -1)

# Compute importance using mean absolute magnitude of weighted ensemble
mean_abs = np.mean(np.abs(mfw_flat), axis=0)

# Indices of top 4 coefficients
top4_idx = np.argsort(mean_abs)[-4:][::-1]

# feature_names = library.get_feature_names()
# # Expand feature names for x,y,z components
expanded_names = ["Coeff 1", "Coeff 2", "Coeff 3", "Coeff 4"]

top4_labels = [expanded_names[i] for i in range(4)]

# Plot 2x2 histograms
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.8))
axs = axs.ravel()

for ax, idx, label in zip(axs, top4_idx, top4_labels):

    # Plot histograms (same colors as before)
    ax.hist(mf_flat[:, idx],  bins=200,color=c_u, alpha=0.35)
    ax.hist(mfw_flat[:, idx], bins=200,color=c_w, alpha=0.35)

    # Compute median and set window ± 20
    med = np.median(mfw_flat[:, idx])

    ax.set_title(label, fontsize=14)
    ax.set_yticks([])
    ax.set_xlabel("Coefficient value")
# Shared legend (same style as forecast plot)
fig.legend(
    [
        plt.Line2D([], [], color=c_u, lw=2),
        plt.Line2D([], [], color=c_w, lw=2),
    ],
    ["Unweighted", "Weighted"],
    loc="upper center",
    ncol=2,
    frameon=False,
    prop={'size': 16}
)

plt.tight_layout(rect=(0,0,1,0.92))
plt.savefig("real_sindy_coeff_histograms_top4.png", dpi=500, transparent=True)
plt.show()

# # ------------------------------------------------------------
# # FORECAST ENSEMBLE USING REAL COEFFICIENTS
# # ------------------------------------------------------------
# # library must match the one used in training
# library = ps.PolynomialLibrary(degree=2)

# t_pred = np.linspace(0, T_pred, steps_pred)
# u0 = X_hist[-1].reshape(3)

# def forecast_from_coeff_ensemble(C_samples):
#     n = C_samples.shape[0]
#     T = len(t_pred)
#     trajectories = np.full((n, T, 3), np.nan)

#     for i in range(n):
#         coef = C_samples[i]  # (n_terms, 3)
#         print(f"Running sample {i}/{n-1}")

#         # Build model matching the training library
#         model = ps.SINDy(feature_library=library)
#         model.fit(X_hist, t_hist)
#         model.optimizer.coef_ = coef.T

#         try:
#             traj = model.simulate(u0, t_pred)

#             # Check if simulation produced NaNs
#             if np.isnan(traj).any():
#                 print(f"⚠️ Sample {i} produced NaNs — skipping")
#                 continue

#             trajectories[i] = traj

#         except Exception as e:
#             trajectories[i] = trajectories[i-1]
#             print(f"❌ Simulation failed for sample {i}: {e}")
#             # skip and leave that trajectory as NaN
#             continue

#     return trajectories

# # Subsample if many samples exist
# n_draw = min(100, len(C_mf))
# idx_mf  = np.random.choice(len(C_mf),  n_draw, replace=True)
# idx_mfw = np.random.choice(len(C_mfw), n_draw, replace=True)

# traj_mf  = forecast_from_coeff_ensemble(C_mf[idx_mf])
# traj_mfw = forecast_from_coeff_ensemble(C_mfw[idx_mfw])

# # ------------------------------------------------------------
# # MEAN AND IQR BANDWIDTH
# # ------------------------------------------------------------
# def mean_iqr(traj):
#     mean = np.median(traj,axis=0)
#     lo   = np.percentile(traj, 25, axis=0)
#     hi   = np.percentile(traj, 75, axis=0)
#     return mean, lo, hi

# mean_mf, lo_mf, hi_mf = mean_iqr(traj_mf)
# mean_mfw, lo_mfw, hi_mfw = mean_iqr(traj_mfw)

# # ------------------------------------------------------------
# # PLOT
# # ------------------------------------------------------------
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 14,
#     "axes.linewidth": 0.9
# })

# colors = ["#4e79a7", "#a17d4f", "#e15759"]  # x,y,z
# c_u = (0.55,0.55,0.55)       # MF (unweighted)
# from matplotlib.cm import plasma
# c_w = plasma(0.6)           # MF-W (weighted)

# fig, axs = plt.subplots(3,1,figsize=(7.3,6.9), sharex=True)

# for k, var in enumerate(["x","y","z"]):
#     ax = axs[k]

#     # Forecast uncertainty bands
#     ax.fill_between(t_hist[-1] + t_pred, lo_mf[:,k],  hi_mf[:,k],  color=c_u, alpha=0.27)
#     ax.fill_between(t_hist[-1] + t_pred, lo_mfw[:,k], hi_mfw[:,k], color=c_w, alpha=0.27)

#     # Forecast means
#     ax.plot(t_hist[-1] + t_pred, mean_mf[:,k],  color=c_u, lw=1.4)
#     ax.plot(t_hist[-1] + t_pred, mean_mfw[:,k], '--', color=c_w, lw=1.6)

#     # History segment
#     ax.plot(t_hist, X_hist[:,k], 'k--', lw=1.3)

#     ax.axvline(t_hist[-1], color="black", ls=":", lw=1)
#     ax.set_ylabel(f"${var}(t)$")
#     ax.set_yticks([])

# axs[-1].set_xlabel(r"$t$ (s)")

# fig.legend(
#     [
#         plt.Line2D([], [], color=c_u, lw=2),
#         plt.Line2D([], [], color=c_w, lw=2),
#     ],
#     ["Unweighted", "Weighted"],
#     loc="upper center",
#     ncol=2,
#     frameon=False,
#     prop={'size': 20}
# )

# plt.tight_layout(rect=(0,0,1,0.93))
# plt.savefig("real_sindy_forecast_uncertainty.png", dpi=500, transparent=True)
# plt.show()
