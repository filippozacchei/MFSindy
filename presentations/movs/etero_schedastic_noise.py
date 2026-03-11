from manim import *

class HeteroscedasticNoise(Scene):
    def construct(self):
        self.camera.background_color = "#00000000"  # transparent for slides

        # --- Colors ---
        COLOR_SIGMA = ORANGE
        COLOR_TRUE = BLUE_C
        COLOR_EXP = GREEN_C
        COLOR_EPS = GRAY_B

        # -------------------------------------------------------
        # Step 1. Start with x_true (clean signal)
        # -------------------------------------------------------
        eq_true = MathTex(
            r"U_i^{\text{true}}",
            font_size=70
        )
        eq_true[0].set_color(COLOR_TRUE)
        eq_true.move_to(UP * 0.5)

        self.play(Write(eq_true), run_time=1.0)
        self.wait(0.6)

        # -------------------------------------------------------
        # Step 2. Add noise term (ε)
        # -------------------------------------------------------
        eq_with_eps = MathTex(
            r"U_i^{\text{true}}", r"+", r"\epsilon",
            font_size=70
        )
        eq_with_eps[0].set_color(COLOR_TRUE)
        eq_with_eps[2].set_color(COLOR_EPS)
        eq_with_eps.move_to(eq_true)

        self.play(TransformMatchingTex(eq_true, eq_with_eps), run_time=1.0)
        self.wait(0.6)

        # -------------------------------------------------------
        # Step 3. Form full homoscedastic model
        # -------------------------------------------------------
        eq_homo = MathTex(
            r"U_i^{\text{exp}}", r"=", 
            r"U_i^{\text{true}}", r"+", 
            r"\epsilon", r",\quad",
            r"\epsilon", r"\sim", r"\mathcal{N}(0,", r"\sigma^2", r")",
            font_size=70
        )
        eq_homo[0].set_color(COLOR_EXP)
        eq_homo[2].set_color(COLOR_TRUE)
        eq_homo[4].set_color(COLOR_EPS)
        eq_homo[9].set_color(COLOR_SIGMA)
        eq_homo.move_to(UP * 0.5)

        self.play(TransformMatchingTex(eq_with_eps, eq_homo), run_time=1.5)
        self.wait(0.8)

        # Subtle indication of transformation (true → exp)
        self.play(
            Indicate(eq_homo[2], color=COLOR_TRUE, scale_factor=1.1),
            Indicate(eq_homo[0], color=COLOR_EXP, scale_factor=1.1),
            run_time=1.2
        )
        self.wait(0.6)

        # -------------------------------------------------------
        # Step 4. Transition to heteroscedastic model
        # -------------------------------------------------------
        eq_hetero = MathTex(
            r"U_i^{\text{exp}}", r"=", 
            r"U_i^{\text{true}}", r"+", 
            r"\epsilon_i", r",\quad",
            r"\epsilon_i", r"\sim", r"\mathcal{N}(0,", r"\sigma_i^2", r")",
            font_size=70
        )
        eq_hetero[0].set_color(COLOR_EXP)
        eq_hetero[2].set_color(COLOR_TRUE)
        eq_hetero[4].set_color(COLOR_EPS)
        eq_hetero[6].set_color(COLOR_EPS)
        eq_hetero[9].set_color(COLOR_SIGMA)
        eq_hetero.move_to(eq_homo)

        # Smooth transition
        self.play(TransformMatchingTex(eq_homo, eq_hetero), run_time=1.8)
        self.wait(0.6)

        # Subtle emphasis on σ_i^2 (heteroscedasticity)
        self.play(
            Indicate(eq_hetero[9], color=COLOR_SIGMA, scale_factor=1.05),
            run_time=1.2
        )
        self.wait(2.5)
