from manim import *
import numpy as np
from .data.generator import simulate_two_mode


class TwoModeBeamScene(Scene):
    def construct(self):
        # --- Simulation parameters ---
        n_lf, n_hf = 1, 1         # number of sensors per fidelity
        noise_lf, noise_hf = 0.05, 0.001
        scale_amp = 0.6
        L = 6.0                   # beam length (for visualization scaling)

        # --- Generate simulation data (modal displacements & sensors) ---
        t, q,  = simulate_two_mode(T=5.0, fs=500.0)
        q1, q2 = q[:, 0], q[:, 2]

        # Mode shapes (approximated for a cantilever)
        def phi1(x): return np.sin(np.pi * x / (2 * L))
        def phi2(x): return np.sin(3 * np.pi * x / (2 * L))

        # Random sensor positions
        sensor_pos_lf = np.random.uniform(0.1, L - 0.1, n_lf)
        sensor_pos_hf = np.random.uniform(0.1, L - 0.1, n_hf)

        # Compute sensor signals
        def displacement(x): return phi1(x) * q1 + phi2(x) * q2
        signals_lf = [displacement(x) + noise_lf * np.random.randn(len(t)) for x in sensor_pos_lf]
        signals_hf = [displacement(x) + noise_hf * np.random.randn(len(t)) for x in sensor_pos_hf]

        # ValueTracker to control time evolution
        time_tracker = ValueTracker(0.0)

        # --- Left: beam oscillation ---
        beam_points = np.linspace(0, L, 400)
        beam_offset = LEFT * 6.5  # move beam leftwards
        beam_base = Line(beam_offset, beam_offset + RIGHT * 6, color=GRAY_B)
        self.add(beam_base)

        beam_curve = always_redraw(
            lambda: VMobject().set_points_smoothly([
                beam_offset + RIGHT * (x / L) * 6 +
                UP * scale_amp * (
                    phi1(x) * np.interp(time_tracker.get_value(), t, q1) +
                    phi2(x) * np.interp(time_tracker.get_value(), t, q2)
                )
                for x in beam_points
            ]).set_color(BLUE)
        )
        self.add(beam_curve)

        # --- Sensor dots (LF = yellow, HF = red) ---
        def make_sensor_dot(x, color):
            return always_redraw(lambda:
                Dot(color=color, radius=0.07).move_to(
                    beam_offset + RIGHT * (x / L) * 6 +
                    UP * scale_amp * (
                        phi1(x) * np.interp(time_tracker.get_value(), t, q1) +
                        phi2(x) * np.interp(time_tracker.get_value(), t, q2)
                    )
                )
            )

        for x in sensor_pos_lf:
            self.add(make_sensor_dot(x, YELLOW))
        for x in sensor_pos_hf:
            self.add(make_sensor_dot(x, RED))

        # --- Right: sensor signal plots ---
        axes = Axes(
            x_range=[0, t[-1], 1],
            y_range=[-1.0, 1.0, 0.5],
            axis_config={"color": WHITE},
            tips=False
        ).scale(0.5).shift(RIGHT * 4)  # shift rightwards
        self.add(axes)

        colors = [YELLOW] * n_lf + [RED] * n_hf
        all_signals = signals_lf + signals_hf

        signal_lines = [
            always_redraw(lambda j=j:
                axes.plot_line_graph(
                    t[t <= time_tracker.get_value()],
                    all_signals[j][t <= time_tracker.get_value()],
                    line_color=colors[j],
                    add_vertex_dots=False
                )
            )
            for j in range(n_lf + n_hf)
        ]
        for line in signal_lines:
            self.add(line)

        # --- Animate time evolution ---
        self.play(time_tracker.animate.set_value(t[-1]), run_time=10, rate_func=linear)
        self.wait(1)

