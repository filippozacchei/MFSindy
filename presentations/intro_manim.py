from manim import *
import numpy as np


class EnsembleSINDy(Scene):
    def construct(self):
        # global style
        self.camera.background_color = BLACK
        text_color = WHITE

        ########################
        # Scene 1: Title
        ########################
        title_main = Tex(
            r"Ensemble SINDy:",
            r" sparse discovery with uncertainty",
            tex_environment="flushleft",
        ).scale(0.9)
        title_main.to_edge(UP)

        self.play(FadeIn(title_main, shift=DOWN))
        self.wait(0.5)

        # We'll shrink the title and keep it in a corner
        self.play(title_main.animate.scale(0.7).to_corner(UL))

        ########################
        # Scene 2: Data + resampling
        ########################
        # Phase plane axes (top row)
        phase_axes = Axes(
            x_range=(-2, 2, 1),
            y_range=(-2, 2, 1),
            x_length=5,
            y_length=3,
            axis_config={"color": GREY_B},
        )
        phase_axes.to_edge(UP).shift(DOWN * 1.2 + RIGHT * 0.5)

        # synthetic "true" trajectory
        t_vals = np.linspace(0, 10, 400)
        x_vals = np.sin(t_vals) + 0.3 * np.sin(3 * t_vals)
        y_vals = np.cos(t_vals) + 0.3 * np.cos(2 * t_vals)
        true_traj = phase_axes.plot_line_graph(
            x_vals,
            y_vals,
            add_vertex_dots=False,
            stroke_color=GREY_B,
            stroke_width=2,
        )

        data_label = Tex("Data space").scale(0.5).next_to(phase_axes, UP, buff=0.15)

        self.play(Create(phase_axes), FadeIn(data_label))
        self.play(Create(true_traj), run_time=2)

        # Multiple colored subselections (bagging)
        colors = [BLUE_C, GREEN_C, YELLOW_C, RED_C]
        sample_windows = VGroup()
        sample_curves = VGroup()
        n_samples = len(colors)

        for i, c in enumerate(colors):
            # choose a time window
            start_idx = 40 * i
            end_idx = start_idx + 120
            xs = x_vals[start_idx:end_idx]
            ys = y_vals[start_idx:end_idx]
            curve = phase_axes.plot_line_graph(
                xs,
                ys,
                add_vertex_dots=False,
                stroke_color=c,
                stroke_width=3,
            )
            sample_curves.add(curve)

            # rough bounding box for the segment
            seg_points = np.column_stack([xs, ys])
            x_min, y_min = seg_points.min(axis=0)
            x_max, y_max = seg_points.max(axis=0)

            p_min = phase_axes.coords_to_point(x_min, y_min)
            p_max = phase_axes.coords_to_point(x_max, y_max)
            rect = Rectangle(
                width=abs(p_max[0] - p_min[0]) * 1.2,
                height=abs(p_max[1] - p_min[1]) * 1.2,
                stroke_color=c,
            )
            rect.move_to((p_min + p_max) / 2)
            sample_windows.add(rect)

        # animate sampling one by one
        for rect, curve in zip(sample_windows, sample_curves):
            self.play(
                Create(rect),
                Create(curve),
                run_time=0.8,
            )
            self.wait(0.2)

        ########################
        # Scene 3: Ensemble of models (coefficient vectors)
        ########################
        # Middle row: bar vectors for coefficients
        coeff_group = VGroup()
        n_coeffs = 6
        n_models = n_samples

        # random synthetic coefficient patterns (sparse)
        rng = np.random.default_rng(0)
        base_true = np.array([1.0, -0.5, 0.0, 0.8, 0.0, -1.2])

        for i in range(n_models):
            vals = base_true + 0.25 * rng.standard_normal(n_coeffs)
            # sparsify some entries
            mask = rng.random(n_coeffs) < 0.4
            vals[mask] = 0.0

            bars = VGroup()
            for j, v in enumerate(vals):
                h = 0.7 * np.clip(v, -1.5, 1.5)
                bar = Rectangle(
                    width=0.15,
                    height=abs(h),
                    stroke_width=0,
                    fill_opacity=1.0,
                    fill_color=colors[i],
                )
                bar.move_to(
                    np.array([0.18 * (j - (n_coeffs - 1) / 2), np.sign(h) * abs(h) / 2, 0])
                )
                bars.add(bar)

            frame = SurroundingRectangle(
                bars,
                buff=0.15,
                color=GREY_B,
                stroke_width=1,
            )

            vec = VGroup(frame, bars)
            coeff_group.add(vec)

        coeff_group.arrange(RIGHT, buff=0.4)
        coeff_group.next_to(phase_axes, DOWN, buff=0.8)

        coef_label = Tex("Model space (coefficients)").scale(0.5)
        coef_label.next_to(coeff_group, UP, buff=0.15)

        # arrows from each phase-plane sample to each coefficient block
        arrows_down = VGroup()
        for curve, vec in zip(sample_curves, coeff_group):
            a = Arrow(
                start=curve.get_bottom() + DOWN * 0.1,
                end=vec.get_top() + UP * 0.1,
                buff=0,
                stroke_width=1.5,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_down.add(a)

        self.play(
            FadeIn(coeff_group),
            FadeIn(coef_label),
            *[GrowArrow(a) for a in arrows_down],
            run_time=1.5,
        )

        self.wait(0.5)

        ########################
        # Scene 4: Collapse to posterior over a few coefficients
        ########################
        # Fade out individual coefficient vectors into a single “index axis”
        self.play(
            FadeOut(arrows_down),
            coeff_group.animate.scale(0.6).to_corner(LEFT).shift(DOWN * 0.8),
            coef_label.animate.to_corner(LEFT).shift(DOWN * 1.4),
            run_time=1.5,
        )

        # Posterior view on the right
        post_axes = Axes(
            x_range=(-2, 2, 1),
            y_range=(0, 1.5, 0.5),
            x_length=5.0,
            y_length=1.8,
            tips=False,
            axis_config={"color": GREY_B},
        )
        post_axes.next_to(coeff_group, RIGHT, buff=1.5).shift(UP * 0.2)

        post_label = Tex("Coefficient distributions").scale(0.5)
        post_label.next_to(post_axes, UP, buff=0.1)

        self.play(Create(post_axes), FadeIn(post_label))

        # use 3 example coefficients with synthetic samples
        centers = [-1.0, 0.0, 1.0]
        colors_post = [BLUE_C, GREEN_C, YELLOW_C]

        for c, col in zip(centers, colors_post):
            samples = c + 0.4 * rng.standard_normal(80)
            # vertical jitter just for aesthetics
            dots = VGroup(
                *[
                    Dot(
                        point=post_axes.coords_to_point(s, 0.1 + 0.2 * rng.random()),
                        radius=0.025,
                        color=col,
                        fill_opacity=0.7,
                    )
                    for s in samples
                ]
            )
            self.play(FadeIn(dots, lag_ratio=0.05, run_time=1.0))

            # then morph into a KDE-like blurred blob (approximate with a filled curve)
            xs = np.linspace(c - 1.5, c + 1.5, 100)
            ys = np.exp(-0.5 * ((xs - c) / 0.4) ** 2)
            ys = ys / ys.max() * 1.3
            points_up = [post_axes.coords_to_point(x, y) for x, y in zip(xs, ys)]
            points_down = [post_axes.coords_to_point(x, 0) for x in xs[::-1]]

            region = VMobject(fill_color=col, fill_opacity=0.35, stroke_color=col)
            region.set_points_as_corners(points_up + points_down)

            median_line = post_axes.get_vertical_line(
                post_axes.coords_to_point(c, 0),
                line_config={"stroke_color": col, "stroke_width": 2},
            )

            self.play(Transform(dots, region), run_time=1.2)
            self.play(Create(median_line), run_time=0.5)

        self.wait(0.5)

        ########################
        # Scene 5: Forecast + uncertainty
        ########################
        # Bottom row: time series with uncertainty band
        t_fore = np.linspace(0, 4, 200)
        true_ts = np.sin(t_fore) + 0.3 * np.sin(2 * t_fore)

        # ensemble trajectories
        n_ens = 25
        ens_ts = []
        for k in range(n_ens):
            phase_shift = 0.1 * rng.standard_normal()
            scale = 1.0 + 0.05 * rng.standard_normal()
            ens_ts.append(scale * np.sin(t_fore + phase_shift))

        ens_ts = np.array(ens_ts)  # (n_ens, Nt)
        median_ts = np.median(ens_ts, axis=0)
        q10 = np.quantile(ens_ts, 0.1, axis=0)
        q90 = np.quantile(ens_ts, 0.9, axis=0)

        ts_axes = Axes(
            x_range=(0, 4, 1),
            y_range=(-2, 2, 1),
            x_length=7,
            y_length=2.5,
            axis_config={"color": GREY_B},
        )
        ts_axes.to_edge(DOWN).shift(UP * 0.5)

        ts_label = Tex("Forecast space").scale(0.5)
        ts_label.next_to(ts_axes, UP, buff=0.1)

        self.play(Create(ts_axes), FadeIn(ts_label))

        # True trajectory
        true_curve = ts_axes.plot_line_graph(
            t_fore, true_ts, add_vertex_dots=False, stroke_color=WHITE, stroke_width=3
        )
        self.play(Create(true_curve), run_time=1.5)

        # thin lines for a few ensemble members, then fade into band + median
        lines_ens = VGroup()
        for k in range(0, n_ens, max(1, n_ens // 10)):
            line = ts_axes.plot_line_graph(
                t_fore,
                ens_ts[k],
                add_vertex_dots=False,
                stroke_color=BLUE_C,
                stroke_width=1,
                stroke_opacity=0.4,
            )
            lines_ens.add(line)

        self.play(Create(lines_ens), run_time=1.5)

        # Uncertainty band
        band_points_up = [ts_axes.coords_to_point(t, y) for t, y in zip(t_fore, q90)]
        band_points_down = [
            ts_axes.coords_to_point(t, y) for t, y in zip(t_fore[::-1], q10[::-1])
        ]
        band = VMobject(
            fill_color=BLUE_C, fill_opacity=0.3, stroke_color=BLUE_C, stroke_width=0
        )
        band.set_points_as_corners(band_points_up + band_points_down)

        median_curve = ts_axes.plot_line_graph(
            t_fore,
            median_ts,
            add_vertex_dots=False,
            stroke_color=BLUE_C,
            stroke_width=2,
        )

        self.play(Transform(lines_ens, band), run_time=1.2)
        self.play(Create(median_curve), run_time=0.8)
        self.wait(0.5)

        ########################
        # Scene 6: Summary icons
        ########################
        summary_box = Rectangle(
            width=8,
            height=3,
            stroke_color=GREY_B,
            stroke_width=1,
        ).move_to(ORIGIN)

        # small schematic icons: data / coefficients / forecast
        icon_data = Tex("1. Resample data").scale(0.6)
        icon_coef = Tex("2. Sparse ensemble").scale(0.6)
        icon_fore = Tex("3. Prediction + uncertainty").scale(0.6)

        icons = VGroup(icon_data, icon_coef, icon_fore).arrange(RIGHT, buff=1.5)
        icons.move_to(summary_box.get_center())

        self.play(
            FadeOut(phase_axes, true_traj, sample_windows, sample_curves, data_label),
            FadeOut(coeff_group, post_axes, post_label),
            FadeOut(ts_axes, ts_label, true_curve, band, median_curve, lines_ens),
            run_time=1.5,
        )

        self.play(Create(summary_box))
        self.play(FadeIn(icons, lag_ratio=0.2, run_time=1.5))

        tagline = Tex(
            r"Ensemble SINDy = sparse discovery + uncertainty quantification"
        ).scale(0.6)
        tagline.next_to(summary_box, DOWN, buff=0.3)
        self.play(FadeIn(tagline))

        self.wait(2)
