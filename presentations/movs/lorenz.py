from manim import *
import numpy as np

class LorenzTrajectories(Scene):
    def construct(self):
        self.camera.background_color = "#000000"

        # --- Colors & noise setup ---
        colors = ["#8d4af9", "#4ebff7", "#ffc078"]
        noise_levels = [0.0, 0.1, 0.5]
        seeds = [8676, 8089, 435451]
        inits = [(10, -10, 31), (-10.5, 22, 41.3), (0.5, 0.8, 21.5)]

        # --- Lorenz system ---
        def lorenz_step(x, y, z, s=10, r=28, b=8/3, dt=0.001):
            dx = s * (y - x)
            dy = x * (r - z) - y
            dz = x * y - b * z
            return x + dx * dt, y + dy * dt, z + dz * dt

        def generate_trajectory(T=30, dt=0.0005, noise=0.0, seed=0, init=(1, 1, 1)):
            np.random.seed(seed)
            n = int(T / dt)
            x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
            x[0], y[0], z[0] = init
            for i in range(1, n):
                x[i], y[i], z[i] = lorenz_step(x[i - 1], y[i - 1], z[i - 1], dt=dt)
            if noise > 0:
                x += np.random.normal(0, noise, n)
                y += np.random.normal(0, noise, n)
            return np.stack([x, y, z], axis=1)

        # --- Generate base attractor for scaling ---
        base_traj = generate_trajectory(T=50, noise=0.0, seed=1)
        xy = base_traj[:, :2]
        xy -= xy.mean(axis=0)  # center it
        xy /= np.max(np.abs(xy)) * 1.2  # normalize nicely
        scale_factor = 6.0  # overall visible scaling

        # --- Convert points to Manim coordinates ---
        def to_points(traj):
            xy = traj[:, :2]
            xy -= xy.mean(axis=0)
            xy /= np.max(np.abs(xy)) * 1.2
            xy *= scale_factor
            xy[:, 0] *= 2.
            xy[:, 1] *= 0.75
            return [np.array([x, y, 0]) for x, y in xy]

        # --- Add faint background attractor ---
        base_points = to_points(base_traj[::10])
        base_line = VMobject(color=GRAY_D, stroke_opacity=0.25)
        base_line.set_points_smoothly(base_points)
        self.add(base_line)

        # --- Create moving trajectories (as fading dots) ---
        all_dots = []
        traj_data = []

        for col, nl, seed, init in zip(colors, noise_levels, seeds, inits):
            traj = generate_trajectory(T=40, noise=nl, seed=seed, init=init)
            points = to_points(traj[::6])
            traj_data.append((points, col))

            # Initial dot trail
            opacities = np.linspace(0.1, 0.9, 100)
            dots = [Dot(point=points[k], color=col, radius=0.05, fill_opacity=opacities[k])
                    for k in range(100)]
            group = VGroup(*dots)
            all_dots.append(group)
            self.add(group)

        # --- Animate flow: fading trail of last 100 points ---
        steps = 5000
        for i in range(100, steps):
            for k, (points, col) in enumerate(traj_data):
                if i < len(points):
                    segment = points[max(0, i - 500):i]
                    opacities = np.linspace(0.1, 0.9, len(segment))  # gradual fade
                    all_dots[k].become(VGroup(*[
                        Dot(p, color=col, radius=0.05, fill_opacity=opacities[idx])
                        for idx, p in enumerate(segment)
                    ]))
            self.wait(0.02)

        self.wait(2)
