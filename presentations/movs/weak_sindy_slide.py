# weak_sindy_slide.py
from manim import *

class WeakSINDYSlide(Scene):
    def construct(self):
        self.camera.background_color = "#00000000"

        # --- Colors ---
        COLOR_PHI   = BLUE_C
        COLOR_V     = GREEN_B
        COLOR_VDOT  = TEAL_B
        COLOR_G     = PURPLE_B
        COLOR_B     = GOLD_B

        # -------------------------------
        # LEFT COLUMN: Equations
        # -------------------------------
        eq1 = MathTex(r"\dot{x}(t) = f(x(t)), \quad t \in [0,T]", font_size=46)

        eq_integral = MathTex(
            r"\int_0^T \dot{x}(t)\,", r"\phi(t)", r"\,dt",
            r"=", 
            r"\int_0^T f(x(t))\,", r"\phi(t)", r"\,dt",
            font_size=44
        )
        eq_integral[1].set_color(COLOR_PHI)
        eq_integral[5].set_color(COLOR_PHI)

        eq_weak = MathTex(
            r"-\int_0^T x(t)\,", r"\dot{\phi}(t)", r"\,dt",
            r"=", 
            r"\int_0^T f(x(t))\,", r"\phi(t)", r"\,dt",
            font_size=44
        )
        eq_weak[1].set_color(COLOR_PHI)
        eq_weak[5].set_color(COLOR_PHI)

        eq_matrix = MathTex(
            r"\dot{V}", r"X", r"\approx", r"V", r"\Theta(X)", r"w", font_size=46
        )
        eq_matrix[0].set_color(COLOR_B)  # \dot{V}
        eq_matrix[1].set_color(COLOR_B)  # X
        eq_matrix[3].set_color(COLOR_G)  # V
        eq_matrix[4].set_color(COLOR_G)  # Θ(X)

        eq_reg = MathTex(r"b", r"\approx", r"G", r"w", font_size=48)
        eq_reg[0].set_color(COLOR_B)
        eq_reg[2].set_color(COLOR_G)

        left_col = VGroup(eq1, eq_integral, eq_matrix, eq_reg).arrange(
            DOWN, aligned_edge=LEFT, buff=0.6
        )
        left_col.to_edge(LEFT, buff=0.9).shift(UP * 0.5)
        left_col[0].shift(UP*0.5)
        # left_col[1].shift(DOWN*0.5)
        left_col[2].shift(DOWN)
        left_col[3].shift(DOWN*1.35)

        # -------------------------------
        # RIGHT COLUMN: Definitions
        # -------------------------------
        defs1 = VGroup(
            MathTex(
                r"V_{ij}",
                r"=",
                r"\phi_j(t_i)",
                font_size=44,
            ),
            MathTex(
                r"\dot{V}_{ij}",
                r"=",
                r"\dot{\phi}_j(t_i)",
                font_size=44,
            )
        )

        # Apply consistent φ coloring
        for eq in defs1:
            eq.set_color_by_tex(r"\phi", COLOR_PHI)
            eq.set_color_by_tex(r"\dot{\phi}", COLOR_PHI)

        # Arrange vertically, with small spacing
        v_defs = VGroup(*defs1).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        v_defs.next_to(left_col)
        v_defs[0].shift(RIGHT).shift(UP*2.1)
        v_defs[1].shift(RIGHT).shift(UP*1.75)
        
        # --- Definitions for b and G with small top-right annotations ---
        b_eq = MathTex(r"b", r"= \dot{V}X", font_size=46)
        b_eq[0].set_color(COLOR_B)
        b_label = Text("measurement vector", font_size=22, slant=ITALIC, font="Arial")
        b_label.next_to(b_eq, UR, buff=0.15).shift(UP*0.05)

        g_eq = MathTex(r"G", r"= V\Theta(X)", font_size=46)
        g_eq[0].set_color(COLOR_G)
        g_label = Text("design matrix", font_size=22, slant=ITALIC, font="Arial")
        g_label.next_to(g_eq, UR, buff=0.15).shift(UP*0.05)

        # Combine equations and labels
        b_group = VGroup(b_eq, b_label)
        g_group = VGroup(g_eq, g_label)

        # Stack vertically
        defs2 = VGroup(b_group, g_group).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        defs2.next_to(left_col).shift(RIGHT).shift(DOWN*2)

        # -------------------------------
        # Animation
        # -------------------------------
        self.play(Write(eq1), run_time=1.0)
        self.wait(0.4)

        # Integration step
        self.play(Write(eq_integral), run_time=1.0)
        self.wait(0.6)

        # --- Align eq_weak in the same position as eq_integral ---
        eq_weak.move_to(eq_integral)

        # 3. Transform smoothly in place
        self.play(TransformMatchingTex(eq_integral, eq_weak), run_time=1.3)
        self.wait(0.6)

        self.play(FadeIn(v_defs, shift=RIGHT*0.2), run_time=1.0)
        self.wait(0.6)
        
        # Matrix form (appears below)
        self.play(FadeIn(eq_matrix, shift=DOWN * 0.3), run_time=1.0)
        self.wait(0.5)

        # Regression form
        self.play(Write(eq_reg), run_time=0.9)
        self.wait(0.4)

        # Definitions on right
        self.play(FadeIn(defs2, shift=RIGHT * 0.3), run_time=1.2)
        self.wait(2.5)
