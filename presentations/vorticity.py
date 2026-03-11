import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------------
# Parameters
# ----------------------------
nx = ny = 100           # higher resolution for smoother video
Lx = Ly = 2.0 * np.pi   # domain size (square, periodic)
dx = Lx / nx
dy = Ly / ny

nu = 1e-4               # viscosity (smaller -> more turbulent-looking, but stiffer)
dt = 1.0e-2             # time step
nsteps = 30             # substeps between frames
nframes = 400           # number of animation frames

# ----------------------------
# Grid
# ----------------------------
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

# ----------------------------
# Initial vorticity field: many random vortices
# ----------------------------
def random_vortices_field(num_vortices=100, min_sigma=0.15, max_sigma=0.35):
    omega0 = np.zeros_like(X)
    rng = np.random.default_rng(12345)

    for _ in range(num_vortices):
        x0 = rng.uniform(0, Lx)
        y0 = rng.uniform(0, Ly)
        sigma = rng.uniform(min_sigma, max_sigma)
        strength = rng.normal(0.0, 1.0)

        # Use periodic distance
        dxp = np.minimum(np.abs(X - x0), Lx - np.abs(X - x0))
        dyp = np.minimum(np.abs(Y - y0), Ly - np.abs(Y - y0))
        r2 = dxp**2 + dyp**2

        omega0 += strength * np.exp(-r2 / (2 * sigma**2))

    # Normalize magnitude a bit
    omega0 /= np.max(np.abs(omega0)) + 1e-12
    return omega0

omega = random_vortices_field()

# ----------------------------
# Spectral wavenumbers (periodic BCs)
# ----------------------------
kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2
K2[0, 0] = 1.0  # avoid division by zero at k=0

def step(omega, nsubsteps):
    """
    Advance the vorticity field in time using a pseudo-spectral
    vorticity–streamfunction formulation of 2D Navier–Stokes.

    dω/dt + u·∇ω = ν ∇²ω,
    u = (∂ψ/∂y, -∂ψ/∂x), ∇²ψ = -ω.
    """
    for _ in range(nsubsteps):
        omega_hat = np.fft.fftn(omega)

        # Streamfunction from vorticity
        psi_hat = -omega_hat / K2
        psi_hat[0, 0] = 0.0  # zero-mean gauge
        # Velocity field
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        u = np.fft.ifftn(u_hat).real
        v = np.fft.ifftn(v_hat).real

        # Gradients of vorticity
        domega_dx_hat = 1j * KX * omega_hat
        domega_dy_hat = 1j * KY * omega_hat
        domega_dx = np.fft.ifftn(domega_dx_hat).real
        domega_dy = np.fft.ifftn(domega_dy_hat).real

        adv = u * domega_dx + v * domega_dy             # advection
        lap_omega_hat = -K2 * omega_hat                 # Laplacian in Fourier
        lap_omega = np.fft.ifftn(lap_omega_hat).real

        # RK2 stage 1
        rhs = -adv + nu * lap_omega
        omega1 = omega + dt * rhs

        # RK2 stage 2
        omega1_hat = np.fft.fftn(omega1)
        psi1_hat = -omega1_hat / K2
        psi1_hat[0, 0] = 0.0
        u1_hat = 1j * KY * psi1_hat
        v1_hat = -1j * KX * psi1_hat
        u1 = np.fft.ifftn(u1_hat).real
        v1 = np.fft.ifftn(v1_hat).real
        domega1_dx_hat = 1j * KX * omega1_hat
        domega1_dy_hat = 1j * KY * omega1_hat
        domega1_dx = np.fft.ifftn(domega1_dx_hat).real
        domega1_dy = np.fft.ifftn(domega1_dy_hat).real

        adv1 = u1 * domega1_dx + v1 * domega1_dy
        lap_omega1_hat = -K2 * omega1_hat
        lap_omega1 = np.fft.ifftn(lap_omega1_hat).real
        rhs1 = -adv1 + nu * lap_omega1

        omega = omega + 0.5 * dt * (rhs + rhs1)

    return omega

# ----------------------------
# Figure / Animation setup
# ----------------------------
# Full-bleed figure, no axes, dark background
fig = plt.figure(figsize=(6, 6), facecolor="black")
ax = plt.axes([0, 0, 1, 1])  # fill entire figure
ax.set_axis_off()

# Use symmetric limits for nicer color dynamics
vmax = np.max(np.abs(omega))
cax = ax.imshow(
    omega.T,
    origin="lower",
    extent=[0, Lx, 0, Ly],
    cmap="magma",
    interpolation="bilinear",
    vmin=-vmax,
    vmax=vmax,
)

def init():
    cax.set_data(omega.T)
    return (cax,)

def animate(frame):
    global omega
    omega = step(omega, nsteps)
    # Optional re-normalization to keep contrast visually strong
    vmax = np.max(np.abs(omega))
    cax.set_data(omega.T)
    cax.set_clim(-vmax, vmax)
    return (cax,)

# ----------------------------
# Time series at random sensors
# ----------------------------
def simulate_vorticity_time_series(num_sensors=3, seed=0):
    """
    Run the vorticity simulation and record the vorticity at a few
    randomly chosen grid points over time.

    Returns
    -------
    t : (nframes,) array
        Time stamps corresponding to each frame.
    signals : (num_sensors, nframes) array
        Vorticity time series at each sensor.
    sensor_indices : list of (ix, iy)
        Grid indices of the sensors on the (nx, ny) grid.
    """
    rng = np.random.default_rng(seed)

    # Choose random grid points as sensors
    sensor_indices = []
    for _ in range(num_sensors):
        ix = rng.integers(0, nx)
        iy = rng.integers(0, ny)
        sensor_indices.append((ix, iy))

    # Local copy of omega so we don't change the global field used for other animations
    omega_local = omega.copy()

    t = np.arange(nframes) * dt * nsteps
    signals = np.zeros((num_sensors, nframes), dtype=float)

    for k in range(nframes):
        # Record vorticity at each sensor
        for s, (ix, iy) in enumerate(sensor_indices):
            signals[s, k] = omega_local[ix, iy]

        # Advance the flow for the next frame
        omega_local = step(omega_local, nsteps)

    return t, signals, sensor_indices

if __name__ == "__main__":
    # === ONLY RUN WHEN vorticity.py IS EXECUTED DIRECTLY ===
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=nframes,
        interval=30,
        blit=True,
    )

    # Optionally save (comment out while debugging)
    anim.save(
        "vorticity_magma_lf2.mp4",
        writer="ffmpeg",
        dpi=200,
        bitrate=8000,
    )

    plt.show()
