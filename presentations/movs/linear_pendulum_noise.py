from manim import *
import numpy as np

class LinearPendulumNoise(Scene):
    def construct(self):
        self.camera.background_color = "#00000000"  # transparent background

        # --- Parameters ---
        T = 8              # duration (seconds of simulated time)
        dt = 0.02
        omega = 1.0
        n_steps = int(T / dt)

        # --- Colors ---
        COLOR_TRUE = BLUE_C
        COLOR_NOISY = ORANGE
        COLOR_PEND = GRAY_A

        # --- Generate data for the linear pendulum ---
        t = np.linspace(0, T, n_steps)
        x_true = np.sin(omega * t)
        v_true = np.cos(omega * t)
        noise_std = 0.25 * np.abs(x_true)
        noise = np.random.normal(0, noise_std)
        x_noisy = x_true + noise

        # ----------------------------------------------------------
        # LEFT: Pendulum visualization (aligned with signal)
        # ----------------------------------------------------------
        pivot = Dot(ORIGIN, color=WHITE)
        rod_length = 2.5

        bob = Dot(color=COLOR_PEND).move_to(
            [rod_length * np.sin(x_true[0]), -rod_length * np.cos(x_true[0]), 0]
        )
        rod = Line(pivot.get_center(), bob.get_center(), color=COLOR_PEND, stroke_width=4)

        pendulum = VGroup(rod, bob, pivot)
        pendulum.scale(0.9)  # match approximate vertical scale of signal
        pendulum.to_edge(LEFT, buff=1.2)  # move fully to left side
        pendulum.shift(DOWN * 0.3)  # center vertically with signal plot

        # ----------------------------------------------------------
        # RIGHT: Time signal (x vs t)
        # ----------------------------------------------------------
        axes = Axes(
            x_range=[0, T, 2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GRAY_B, "stroke_width": 2},
            tips=False,
        ).scale(0.9)
        axes.to_edge(RIGHT, buff=1.0)  # bring to the right side
        axes.shift(DOWN * 0.3)  # align with pendulum

        label_t = axes.get_x_axis_label("t", direction=RIGHT)
        label_x = axes.get_y_axis_label("x(t)", direction=UP)

        true_curve = VMobject(color=COLOR_TRUE, stroke_width=4)
        noisy_dots = VGroup()
        true_points = [axes.c2p(t[0], x_true[0])]
        true_curve.set_points_as_corners(true_points)

        self.add(pendulum, axes, label_t, label_x, true_curve, noisy_dots)

        # ----------------------------------------------------------
        # Animation loop
        # ----------------------------------------------------------
        for i in range(1, n_steps):
            # --- Pendulum motion ---
            theta = x_true[i]
            bob.move_to(pivot.get_center() + [rod_length * np.sin(theta), -rod_length * np.cos(theta), 0])
            rod.put_start_and_end_on(pivot.get_center(), bob.get_center())

            # --- Update true trajectory ---
            true_points.append(axes.c2p(t[i], x_true[i]))
            true_curve.set_points_as_corners(true_points)

            # --- Add noisy point ---
            noisy_point = Dot(
                axes.c2p(t[i], x_noisy[i]),
                color=COLOR_NOISY,
                radius=0.035,
                fill_opacity=0.9,
            )
            noisy_dots.add(noisy_point)
            self.add(noisy_point)

            self.wait(0.02)

        self.wait(2)
