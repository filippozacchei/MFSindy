import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def lax_wendroff_step(u, dx, dt):
    # flux
    f = 0.5 * u**2
    
    # compute derivative
    du = np.roll(u, -1) - np.roll(u, 1)
    df = np.roll(f, -1) - np.roll(f, 1)

    return u - dt/(2*dx) * df + (dt**2)/(2*dx**2) * (u * du)

def simulate_lw(N, t_end, dt):
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    dx = x[1] - x[0]
    u = np.sin(x)

    frames = []
    t = 0.0
    while t < t_end:
        frames.append(u.copy())
        u = lax_wendroff_step(u, dx, dt)
        t += dt

    return x, np.array(frames)


# --------------------------------------------------------
# Time parameters (SAME dt for both grids)
# --------------------------------------------------------
N_coarse = 400
N_fine   = 4000
t_end = 4.0

# stable dt (based on fine grid CFL)
dx_fine = (2*np.pi)/N_fine
dt = 0.4 * dx_fine / 1.0   # max |u| ~ 1 initially

x_coarse, U_coarse = simulate_lw(N_coarse, t_end, dt)
x_fine,   U_fine   = simulate_lw(N_fine,   t_end, dt)

n_frames = min(len(U_coarse), len(U_fine))//2

# --------------------------------------------------------
# Animation
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,4))
line1, = ax.plot([], [], '-', label=f'Coarse N={N_coarse}')
line2, = ax.plot([], [], '-',  label=f'Fine N={N_fine}')
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_title('Burgers Equation – Lax–Wendroff (Dispersion Visible on Coarse Grid)')
ax.legend()

def update(frame):
    line1.set_data(x_coarse, U_coarse[frame])
    line2.set_data(x_fine,   U_fine[frame])
    return line1, line2

ani = FuncAnimation(fig, update, frames=n_frames, interval=30)
plt.show()
