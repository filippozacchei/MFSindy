# weak_sindy_slide.py
from manim import *

class SINDYSlide(Scene):
    def construct(self):
        # Transparent background (controlled by CLI flag --transparent)
        self.camera.background_color = "#00000000"
        
        # --- Define colors for conceptual consistency ---
        COLOR_THETA = BLUE_C
        COLOR_W = GREEN_C
        COLOR_LAMBDA = PURPLE_C

        # --- Left column: equations ---
        eq1 = MathTex(
            r"\dot{x} = f(x), \quad t > 0",
            font_size=46
        )

        # --- Equation 2 ---
        eq2 = MathTex(
            r"\dot{x} \approx ",
            r"\Theta(x)",
            r"\mathbf{W}",
            font_size=46
        )
        # Apply colors only to specific parts
        eq2[1].set_color(COLOR_THETA)  # Θ(x)
        eq2[2].set_color(COLOR_W)      # W

        # --- Equation 3 ---
        eq3 = MathTex(
            r"\mathbf{W}",
            r" = \arg\min_{W} \Big( \|\dot{X} - ",
            r"\Theta(X)",
            r"\mathbf{W}",
            r"\|_2^2 + ",
            r"\lambda\|W\|_0",
            r"\Big)",
            font_size=40
        )
        # Color only the terms you need
        eq3[0].set_color(COLOR_W)         # W (left-hand side)
        eq3[2].set_color(COLOR_THETA)     # Θ(X)
        eq3[3].set_color(COLOR_W)         # W (right-hand side)
        eq3[5].set_color(COLOR_LAMBDA)    # λ‖W‖₀

        equations = VGroup(eq1, eq2, eq3).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        equations.to_edge(LEFT).shift(UP * 0.8)

        # --- Right column: annotations ---
        theta_label = MathTex(r"\Theta(x):", font_size=36, color=COLOR_THETA)
        theta_text = Text("library of candidate functions", font_size=26)
        w_label = MathTex(r"\mathbf{W}:", font_size=36, color=COLOR_W)
        w_text = Text("sparse coefficients", font_size=26)
        lambda_label = MathTex(r"\lambda\|W\|_0:", font_size=36, color=COLOR_LAMBDA)
        lambda_text = Text("sparsity regularization", font_size=26)

        annotations = VGroup(
            VGroup(theta_label, theta_text).arrange(RIGHT, buff=0.25),
            VGroup(w_label, w_text).arrange(RIGHT, buff=0.25),
            VGroup(lambda_label, lambda_text).arrange(RIGHT, buff=0.25)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        annotations.next_to(equations, RIGHT, buff=0.5).shift(UP * 0.3)

        # --- Animation sequence ---
        # 1. Appear the base system equation
        self.play(Write(eq1), run_time=1.0)
        self.wait(0.6)

        # 2. Morph smoothly into SINDy approximation
        self.play(
            TransformFromCopy(eq1, eq2, path_arc=-PI / 2),
            run_time=1.6
        )
        self.wait(0.5)

        # 3. Show Θ(x) annotation, highlight term
        self.play(FadeIn(annotations[0], shift=RIGHT * 0.2), run_time=0.8)
        self.wait(0.4)

        # 4. Show W annotation, highlight term
        self.play(FadeIn(annotations[1], shift=RIGHT * 0.2), run_time=0.8)
        self.wait(0.4)

        # 5. Transition to sparse regression formulation (smooth zoom-in)
        self.play(
            TransformFromCopy(eq2, eq3, path_arc=PI / 2),
            run_time=1.8
        )
        self.wait(0.5)

        # 6. Show λ‖W‖₀ annotation
        self.play(FadeIn(annotations[2], shift=RIGHT * 0.2), run_time=0.8)

        # 7. Hold for discussion
        self.wait(2.5)
