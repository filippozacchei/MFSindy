from manim import *


class STLSQ_SINDy_Intro(Scene):
    def construct(self):
        # ------------------------------------------------------------
        # 1. SINDy model: Udot = Θ(U) Ξ
        # ------------------------------------------------------------
        sindy_eq = MathTex(
            r"\dot{U} \approx \Theta(U)\,\Xi",
            font_size=60,
        )

        self.play(Write(sindy_eq))
        self.wait(1.5)

        # ------------------------------------------------------------
        # 2. Sparse regression objective (STLSQ target problem)
        # ------------------------------------------------------------
        objective_eq = MathTex(
            r"\Xi = \arg\min_{\tilde{\Xi}} \left(",
            r"\|\dot{U} - \Theta(U)\tilde{\Xi}\|_2^2",
            r" + \lambda \|\tilde{\Xi}\|_0",
            r"\right)",
            font_size=60,
        )
        objective_eq.next_to(sindy_eq, DOWN, buff=0.8)

        self.play(Write(objective_eq))
        self.wait(1.5)

        # ------------------------------------------------------------
        # 3. Show scaling of the L2 term: S \dot{U}, S Θ(U)
        # ------------------------------------------------------------
        objective_scaled = MathTex(
            r"\Xi = \arg\min_{\tilde{\Xi}} \left(",
            r"(\dot{U} - \Theta(U)\tilde{\Xi})^\top(\dot{U} - \Theta(U)\tilde{\Xi})",
            r" + \lambda \|\tilde{\Xi}\|_0",
            r"\right)",
            font_size=60,
        )
        objective_scaled.move_to(objective_eq)

        # Transform objective into its scaled form
        self.play(
            TransformMatchingTex(objective_eq, objective_scaled)
        )
        self.wait(0.5)

        # Highlight the scaled L2 term (only place where we use colour)
        l2_term = objective_scaled[1]  # the middle piece: ‖S U̇ - SΘ(U)Ξ̃‖²₂
        self.play(
            l2_term.animate.set_color(YELLOW).scale(1.05)
        )
        self.wait(2)
