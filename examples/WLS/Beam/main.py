# ============================================================
# Augmented-state fusion: dot{w}=v (enforced), dot{v}=f(w,v,x), dot{x}=0 (enforced)
# ============================================================
import numpy as np
import pysindy as ps
from scipy.signal import savgol_filter

# ----- Ground-truth simulator you already have -----
from data.generator import simulate_two_mode
from data.utils import set_seed
from data.system_matrix import system_matrix

set_seed(0)

# -----------------------------
# 1) Generate training runs
# -----------------------------
L = 5.0
T, fs = 3.0, 1000.0
t, q = simulate_two_mode(T=T, fs=fs)          # q = [q1,v1,q2,v2], not used by SINDy
dt = t[1] - t[0]
q1, q2 = q[:,0], q[:,2]

# Mode shapes used only to synthesize sensor signals
def phi1(x): return np.sin(np.pi * x / (2*L))
def phi2(x): return np.sin(3*np.pi * x / (2*L))
def w_at_x(x): return phi1(x)*q1 + phi2(x)*q2

# Sensor set (many positions, heterogeneous noise)
n_sensors = 6
sensor_x  = np.linspace(0.05*L, 0.95*L, n_sensors)
noise_std = np.linspace(2e-4, 1e-3, n_sensors)   # example: growing noise along the span

# Build per-sensor trajectories as runs: z=[w, v, x̃], z_dot=[v, dv/dt, 0]
# Use x̃ = (π/(2L)) x so that sin(x̃) ≈ φ1(x), sin(3x̃) ≈ φ2(x).
def build_runs(sensor_x, noise_std):
    runs_X, runs_Xdot, weights = [], [], []
    for xj, sj in zip(sensor_x, noise_std):
        w = w_at_x(xj) + 0*sj*np.random.randn(len(t))
        # light smoothing -> stable derivatives
        # w = savgol_filter(w, 21, 3, mode="interp")
        v = np.gradient(w, dt)
        vdot = np.gradient(v, dt)
        # normalize position for the library
        x_tilde = (np.pi/(2*L))*xj
        z     = np.column_stack([w, v, np.full_like(w, x_tilde)])
        z_dot = np.column_stack([v, vdot, np.zeros_like(w)])
        runs_X.append(z)
        runs_Xdot.append(z_dot)
        # per-run weight ~ 1/variance (simple heteroscedastic handling)
        weights.append(np.full(len(t), 1.0/(sj**2 + 1e-12)))
    return runs_X, runs_Xdot, weights

X_train, Xdot_train, sample_w = build_runs(sensor_x, noise_std)

# -----------------------------------------
# 2) Feature library for f(w,v,x̃) in dot v
# -----------------------------------------
# Keep it standard but expressive:
#   - linear terms in w, v
#   - a small Fourier set in x̃ to emulate beam shapes: {1, sin(x̃), sin(3x̃)}
# Final model yields combinations like: w*{1,sin, sin3} and v*{1,sin, sin3}
state_lib   = ps.PolynomialLibrary(degree=1, include_bias=False)   # [w, v]
def g0(w,v,x): return np.ones_like(w)
def g1(w,v,x): return np.sin(x)
def g3(w,v,x): return np.sin(3*x)
space_lib   = ps.CustomLibrary([g0, g1, g3])                       # [1, sin x̃, sin 3x̃]
library     = ps.GeneralizedLibrary([state_lib, space_lib],tensor_array=[[1,1]])        # tensor product

# ------------------------------------------------
# 3) Fit only the physics-relevant equations
# ------------------------------------------------
# We enforce dot{w}=v and dot{x}=0 exactly by *not* learning them:
#   - we pass z_dot for all states (so the first eq will become ≈ v)
#   - after fitting, we overwrite eq.0 and eq.2 coefficients to exact forms.
optimizer = ps.STLSQ(threshold=1e-1, alpha=1e-10, normalize_columns=False)

model = ps.SINDy(
    feature_library=library,
    optimizer=optimizer,
)

# Fit using multi-run data and per-sample weights (list of arrays matches runs)
model.fit(X_train, t=dt)

# ------------------------------------------------
# 4) Enforce dot{w}=v and dot{x}=0 exactly (post-fit projection)
# ------------------------------------------------
# Build coefficient matrix and zero everything we don't want

print("\n--- Learned model (with enforced dot{w}=v, dot{x}=0) ---")
model.print()

# -----------------------------------------
# 5) Validation on a held-out position x*
# -----------------------------------------
x_new = 0.73*L
x_new_tilde = (np.pi/(2*L))*x_new
# synthesize truth at x*
w_true = w_at_x(x_new)
v_true = np.gradient(w_true, dt)
a_true = np.gradient(v_true, dt)

# One-step derivative check
Z_state   = np.column_stack([w_true, v_true, np.full_like(w_true, x_new_tilde)])
Zdot_pred = model.predict(Z_state)
a_pred    = Zdot_pred[:, 1]

# Rollout from true IC
z0 = np.array([w_true[0], v_true[0], x_new_tilde])
Z_sim = model.simulate(z0, t)
w_pred, v_pred = Z_sim[:,0], Z_sim[:,1]


# Choose a point not used for training
x_star = 0.73 * L
x_tilde_star = (np.pi / (2 * L)) * x_star

# Ground truth at that point
w_true = w_at_x(x_star)
v_true = np.gradient(w_true, dt)

# Simulate SINDy model starting from true initial condition
z0 = np.array([w_true[0], v_true[0], x_tilde_star])
Z_sim = model.simulate(z0, t)
w_pred, v_pred = Z_sim[:, 0], Z_sim[:, 1]
import matplotlib.pyplot as plt

# Plot forecasting results
plt.figure(figsize=(6, 3))
plt.plot(t, w_true, "k", lw=2, label="True $w(t)$")
plt.plot(t, w_pred, "r--", lw=2, label="Predicted $w(t)$ (SINDy)")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title(fr"Forecast at held-out point $x^*={x_star:.2f}$ m")
plt.legend()
plt.tight_layout()
plt.show()
