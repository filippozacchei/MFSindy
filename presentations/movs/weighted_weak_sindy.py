# weighted_weak_sindy_noise_slide.py
from manim import *

class WeightedWeakSINDYNoise(Scene):
    def construct(self):
        self.camera.background_color = "#00000000"

        # --- COLORS ---
        COLOR_X = GOLD_B
        COLOR_V = GREEN_B
        COLOR_VDOT = TEAL_B
        COLOR_G = PURPLE_B
        COLOR_SIGMA = BLUE_B
        COLOR_W = ORANGE

        # ----------------------------------------
        # NOISY DATA MODEL
        # ----------------------------------------
        noise_label = Text(
            "Noisy data model (heteroskedastic)",
            font_size=28,
            weight=BOLD,
            color=WHITE,
            font="Arial"
        )
        noise_eq = MathTex(
            r"x_i^{\text{exp}} = x_i^{\text{true}} + \eta_i, \quad "
            r"\mathbb{E}[\eta_i]=0, \;"
            r"\mathrm{Var}(\eta_i)=",
            r"\sigma_i",
            r"^2",
            font_size=42
        )
        noise_eq.set_color_by_tex(r"\sigma_i", COLOR_SIGMA)

        noise_group = VGroup(noise_label, noise_eq).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        # ----------------------------------------
        # WEAK FORMULATION
        # ----------------------------------------
        weak_label = Text(
            "Weak formulation",
            font_size=28,
            weight=BOLD,
            color=WHITE,
            font="Arial"
        )
        weak_eq = MathTex(
            r"b = ",
            r"\dot{V}",
            r"x^{\text{exp}}",
            r", \quad "
            r"G",
            r" = ",
            r"V",
            r"\,\Theta(",
            r"x^{\text{exp}}",
            r"), \quad "
            r"r = b - ",
            r"G",
            r"w",
            font_size=42
        )
        weak_eq.set_color_by_tex(r"\dot{V}", COLOR_VDOT)
        weak_group = VGroup(weak_label, weak_eq).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        # ----------------------------------------
        # COVARIANCE STRUCTURE
        # ----------------------------------------
        cov_label = Text(
            "Covariance structure",
            font_size=28,
            weight=BOLD,
            color=WHITE,
            font="Arial"
        )
        cov_eq = MathTex(
            r"\Sigma_b",
            r" = ",
            r"\dot{V}",
            r"\,\mathrm{diag}(\sigma_i^2)\,",
            r"\dot{V}",
            r"^\top,"
            r"\quad ",
            r"\Sigma_G",
            r" = V(\mathrm{Cov}(\Theta(X)))V^\top,"
            r"\quad ",
            r"\Sigma_r",
            r" = ",
            r"\Sigma_b",
            r" + \Sigma_G",
            font_size=40
        )
        cov_eq.set_color_by_tex(r"\dot{V}", COLOR_VDOT)
        cov_eq.set_color_by_tex(r"\Sigma_b", COLOR_SIGMA)

        cov_group = VGroup(cov_label, cov_eq).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        # ----------------------------------------
        # ASSUMPTION
        # ----------------------------------------
        assumption_label = Text(
            "Simplifying assumption",
            font_size=28,
            weight=BOLD,
            color=WHITE,
            font="Arial"
        )
        assumption_eq = MathTex(
            r"\Sigma_r \approx \Sigma_b",
            font_size=42,
            color=WHITE
        )

        assumption_group = VGroup(assumption_label, assumption_eq).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        # ----------------------------------------
        # GLS
        # ----------------------------------------
        gls_label = Text(
            "Generalized Least Squares (GLS)",
            font_size=28,
            weight=BOLD,
            color=WHITE,
            font="Arial"
        )
        gls_eq = MathTex(
            r"\widehat{w} = (",
            r"G",
            r"^\top \Sigma_r^{-1} ",
            r"G",
            r")^{-1} ",
            r"G",
            r"^\top \Sigma_r^{-1} b"
            r"\;\approx\;"
            r"(",
            r"G",
            r"^\top ",
            r"\Sigma_b",
            r"^{-1} ",
            r"G",
            r")^{-1} ",
            r"G",
            r"^\top ",
            r"\Sigma_b",
            r"^{-1} b",
            font_size=40
        )
        gls_eq.set_color_by_tex(r"G", COLOR_G)
        gls_eq.set_color_by_tex(r"\Sigma_b", COLOR_SIGMA)

        gls_group = VGroup(gls_label, gls_eq).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        # ----------------------------------------
        # Stack all vertically
        # ----------------------------------------
        all_groups = VGroup(
            noise_group,
            weak_group,
            cov_group,
            assumption_group,
            gls_group
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        all_groups.to_edge(LEFT, buff=1).shift(DOWN*0.3)

        # ----------------------------------------
        # ANIMATION SEQUENCE
        # ----------------------------------------
        self.play(FadeIn(noise_group, shift=DOWN*0.2), run_time=1.2)
        self.wait(0.3)
        self.play(FadeIn(weak_group, shift=DOWN*0.2), run_time=1.0)
        self.wait(0.3)
        self.play(FadeIn(cov_group, shift=DOWN*0.2), run_time=1.0)
        self.wait(0.3)
        self.play(FadeIn(assumption_group, shift=DOWN*0.2), run_time=0.9)
        self.wait(0.3)
        self.play(FadeIn(gls_group, shift=DOWN*0.2), run_time=1.0)
        self.wait(3)
