from manim import *


class GLSAnimation(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        # -------------------------------
        # 1. Original OLS / ridge objective
        # -------------------------------
        eq_ols = MathTex(
            r"{\Xi}",
            r"=",
            r"\operatorname{argmin}_{\widehat{\Xi}}",
            r"\left\{",
            r"(V\Theta\widehat{\Xi}-\dot{V}\mathbf{U})^{T}",
            r"(V\Theta\widehat{\Xi}-\dot{V}\mathbf{U})",
            r"+",
            r"\lambda\|\widehat{\Xi}\|_{2}^{2}",
            r"\right\}",
            color=WHITE,
        ).scale(0.9)
        eq_ols.to_edge(UP)

        label_ols = Tex(
            "Ordinary least squares (ridge-regularized)",
            color=GREY_B,
        ).scale(0.6)
        label_ols.next_to(eq_ols, 3*DOWN)

        self.play(Write(eq_ols), run_time=2)
        self.play(FadeIn(label_ols, shift=1.5*DOWN))
        self.wait(2)

        # Highlight the data misfit term (G w - b)^T (G w - b)
        misfit_group = VGroup(eq_ols[4], eq_ols[5])
        misfit_box = SurroundingRectangle(misfit_group, color=YELLOW, buff=0.1)

        misfit_label = Tex(
            "unweighted data misfit",
            color=YELLOW,
        ).scale(0.5)
        misfit_label.next_to(misfit_box, 1.5*DOWN)

        self.play(Create(misfit_box), FadeIn(misfit_label), run_time=1.2)
        self.wait(0.7)

        # -------------------------------
        # 2. Transform into GLS objective
        # -------------------------------
        eq_gls = MathTex(
            r"{\Xi}",
            r"=",
            r"\operatorname{argmin}_{\widehat{\Xi}}",
            r"\left\{",
            r"(V\Theta\widehat{\Xi}-\dot{V}\mathbf{U})^{T}",
            r"\Sigma^{-1}",
            r"(V\Theta\widehat{\Xi}-\dot{V}\mathbf{U})",
            r"+",
            r"\lambda\|\widehat{\Xi}\|_{2}^{2}",
            r"\right\}",
            color=WHITE,
        ).scale(0.9)
        eq_gls.move_to(eq_ols)

        label_gls = Tex(
            "Generalized least squares (GLS)",
            color=GREY_B,
        ).scale(0.6)
        label_gls.next_to(eq_gls, 3*DOWN)

        # Remove the misfit highlight
        self.play(FadeOut(misfit_box), FadeOut(misfit_label), run_time=0.6)

        # Transform equation AND replace caption in one go
        self.play(
            TransformMatchingTex(eq_ols, eq_gls),
            ReplacementTransform(label_ols, label_gls),
            run_time=2,
        )
        self.wait(0.5)

        # -------------------------------
        # 3. Highlight W and explain
        # -------------------------------
        W_part = eq_gls.get_part_by_tex(r"\Sigma^{-1}")
        W_box = SurroundingRectangle(W_part, color=BLUE, buff=0.1)

        W_text = Tex(
            r"$\Sigma$: covariance matrix ",
            tex_environment="flushleft",
            color=BLUE,
        ).scale(0.55)
        W_text.next_to(eq_gls, 1*DOWN)

        self.play(Create(W_box), FadeIn(W_text, shift=DOWN), run_time=1.5)

        # Final layout
        final_group = VGroup(eq_gls, label_gls, W_box, W_text)
        # self.play(final_group.animate.shift(DOWN * 0.3), run_time=0.8)
        self.wait(2)
