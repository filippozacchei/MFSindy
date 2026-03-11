from manim import *
import numpy as np

from vorticity import simulate_vorticity_time_series

class VorticitySensorSignals(Scene):
    def construct(self):
        # --- get clean data from the simulation ---
        t, signals, sensor_indices = simulate_vorticity_time_series(
            num_sensors=3,
            seed=1
        )
        nframes = len(t)
        num_sensors = signals.shape[0]
        T_max = float(t[-1])

        # --- add three different noise levels ---
        rng = np.random.default_rng(0)
        # interpret as: sensor 1 = high fidelity, sensor 2 = medium, sensor 3 = low
        noise_scales = np.array([0.6, 0.2, 0.01])  # relative amplitudes
        base_std = np.std(signals)

        noisy_signals = np.zeros_like(signals)
        for s in range(num_sensors):
            noise = noise_scales[s] * base_std * rng.standard_normal(size=nframes)
            noisy_signals[s, :] = signals[s, :] + noise

        # --- symmetric y-range so x-axis is in the middle ---
        y_abs = 1.1 * np.max(np.abs(noisy_signals))
        ymin, ymax = -y_abs, y_abs

        # --- axes setup ---
        axes = Axes(
            x_range=[0, T_max, T_max / 5],
            y_range=[ymin, ymax, (ymax - ymin) / 4],
            x_length=10,
            y_length=5,
            tips=False,
        )
        axes.shift(DOWN * 0.5)

        x_label = axes.get_x_axis_label("t")
        y_label = axes.get_y_axis_label(r"\omega(t)")
        labels = VGroup(x_label, y_label)

        self.play(Create(axes), Write(labels))

        # --- time tracker ---
        time_tracker = ValueTracker(0.0)

        colors = [GREEN, YELLOW, RED]  # e.g. low, medium, high noise
        sensor_curves = VGroup()

        # For each sensor, build an always_redraw curve that extends with time
        for s in range(num_sensors):
            def make_curve(sensor_index=s):
                def updater():
                    current_t = time_tracker.get_value()
                    idx = np.searchsorted(t, current_t, side="right")
                    if idx < 2:
                        return axes.plot_line_graph(
                            [0, 1e-6],
                            [0, 0],
                            add_vertex_dots=False,
                            line_color=colors[sensor_index],
                            stroke_width=3,
                        )
                    return axes.plot_line_graph(
                        t[:idx],
                        noisy_signals[sensor_index, :idx],
                        add_vertex_dots=False,
                        line_color=colors[sensor_index],
                        stroke_width=3,
                    )
                return always_redraw(updater)

            curve = make_curve()
            sensor_curves.add(curve)

        # vertical line indicating current time
        time_line = always_redraw(
            lambda: axes.get_vertical_line(
                axes.c2p(time_tracker.get_value(), 0),
                color=BLUE,
                stroke_width=2,
            )
        )

        # --- legend: ω_{1}(t), ω_{2}(t), ω_{3}(t) ---
        legend_items = VGroup()
        for s in range(num_sensors):
            sensor_label = MathTex(
                rf"\omega_{{{s+1}}}(t)",
                color=colors[s]
            )
            sensor_label.scale(0.6)
            legend_items.add(sensor_label)

        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        legend_dots = VGroup()
        for s in range(num_sensors):
            dot = Dot(color=colors[s], radius=0.06)
            dot.next_to(legend_items[s], LEFT, buff=0.15)
            legend_dots.add(dot)

        legend = VGroup(legend_dots, legend_items)
        legend.arrange(RIGHT, buff=0.2)
        legend.next_to(axes, UP + RIGHT, buff=0.4)

        # add all time-varying elements
        self.add(*sensor_curves, time_line, legend)

        # animate time from 0 to T_max
        self.play(
            time_tracker.animate.set_value(T_max),
            run_time=10,  # total video duration
            rate_func=linear,
        )

        self.wait(1)
        
from manim import *
import numpy as np

from vorticity import simulate_vorticity_time_series


from manim import *
import numpy as np

from vorticity import simulate_vorticity_time_series


class VorticityConditionDependent(Scene):
    def construct(self):
        # --- clean data from one sensor ---
        t, signals, sensor_indices = simulate_vorticity_time_series(
            num_sensors=1,
            seed=2,
        )
        clean = signals[0]
        nframes = len(t)
        T_max = float(t[-1])

        rng = np.random.default_rng(1)
        base_std = np.std(clean)

        # --- magnitude-dependent noise ---
        amp = np.abs(clean)
        amp_norm = amp / (np.max(amp) + 1e-12)

        low_noise = 0.1
        high_noise = 0.8
        noise_factor = low_noise + high_noise * amp_norm

        noise = noise_factor * base_std * rng.standard_normal(size=nframes)
        noisy = clean + noise

        # --- symmetric y-range ---
        y_abs = 1.1 * max(np.max(np.abs(clean)), np.max(np.abs(noisy)))
        ymin, ymax = -y_abs, y_abs

        axes = Axes(
            x_range=[0, T_max, T_max / 5],
            y_range=[ymin, ymax, (ymax - ymin) / 4],
            x_length=10,
            y_length=5,
            tips=False,
        )
        axes.shift(DOWN * 0.5)

        x_label = axes.get_x_axis_label("t")
        y_label = axes.get_y_axis_label(r"\omega(t)")
        labels = VGroup(x_label, y_label)
        self.play(Create(axes), Write(labels))

        # clean background curve
        clean_curve_full = axes.plot_line_graph(
            x_values=t,
            y_values=clean,
            add_vertex_dots=False,
            line_color=GREY_B,
            stroke_width=2,
        )
        self.play(Create(clean_curve_full), run_time=1.5)

        time_tracker = ValueTracker(0.0)

        def noisy_curve_updater():
            current_t = time_tracker.get_value()
            idx = np.searchsorted(t, current_t, side="right")
            if idx < 2:
                return axes.plot_line_graph(
                    [0, 1e-6],
                    [noisy[0], noisy[0]],
                    add_vertex_dots=False,
                    line_color=YELLOW,
                    stroke_width=3,
                )
            return axes.plot_line_graph(
                t[:idx],
                noisy[:idx],
                add_vertex_dots=False,
                line_color=YELLOW,
                stroke_width=3,
            )

        noisy_curve = always_redraw(noisy_curve_updater)

        time_line = always_redraw(
            lambda: axes.get_vertical_line(
                axes.c2p(time_tracker.get_value(), 0),
                color=BLUE,
                stroke_width=2,
            )
        )

        self.add(noisy_curve, time_line)

        self.play(
            time_tracker.animate.set_value(T_max),
            run_time=10,
            rate_func=linear,
        )

        self.wait(1)
