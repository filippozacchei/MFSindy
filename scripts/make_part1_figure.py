"""Rebuild the Part I results figure from the committed error tables.

The figure in the paper was assembled in Keynote from a run that predates
deduplication, the coverage-2 support rule and per-rung tuning, and its numbers
no longer match anything the code produces: it reports Lorenz WMF 6.23e-4 and HF
9.79e-3 where the current tables give 1.15e-2 and 2.16. This regenerates the
quantitative columns from the tables themselves, so the figure and the results
cannot drift apart again.

It also shows the ladder Referee 2 asked for rather than four rungs: the two
single-fidelity baselines, naive pooling, the pooled weak-SINDy weighting, plain
inverse-variance group weighting, and the proposed weighting. The referee's
question is whether the last beats the two before it, which the old four-rung
figure could not address.

Canvas matches the original 1920x1080 pt so the result drops into the same
layout; the trajectory thumbnails in the original's first column are not
reproduced here and are left to be re-pasted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/Users/filippozacchei/COde/Multi_Fidelity_SINDy__Sparse_Discovery_of_"
           "Nonlinear_Dynamical_Systems_with_Fidelity_Weighted_Measurements/Figure")

#: The ladder, in the order the argument runs: single fidelity, naive pooling,
#: the two simpler weightings a referee would reach for, then ours.
RUNGS = ["HF", "LF", "MF", "PMF", "VMF", "MF_w"]
LABELS = {
    "HF": "HF", "LF": "LF", "MF": "MF",
    "PMF": "MF-P", "VMF": "MF-IV", "MF_w": "WMF",
}
COLORS = {
    "HF": "#4C72B0", "LF": "#DD8452", "MF": "#55A868",
    "PMF": "#8172B3", "VMF": "#937860", "MF_w": "#C44E52",
}

CASES = [
    ("Lorenz", "lorenz", "lorenz_part1_errors.csv",
     r"$\dot{x}=\sigma(y-x)$" "\n" r"$\dot{y}=x(\rho-z)-y$" "\n" r"$\dot{z}=xy-\beta z$"),
    ("Pendulum", "pendulum", "pendulum_part1_errors.csv",
     r"$\dot{\theta}=\omega$" "\n" r"$\dot{\omega}=c\,\omega-\frac{g}{\ell}\theta$"),
    ("Hopf", "hopf", "hopf_part1_errors.csv",
     r"$\dot{r}=\mu r-r^{3}$" "\n" r"$\dot{\theta}=\omega$"),
    ("Burgers", "burgers", "burgers_part1_errors.csv",
     r"$u_t=u\,u_x+\nu\,u_{xx}$"),
    ("Navier–Stokes", "isothermal_flow", "navierstokes_part1_errors.csv",
     r"$\mathbf{u}_t=-(\mathbf{u}\cdot\nabla)\mathbf{u}$" "\n"
     r"$\quad-\rho^{-1}(\nabla p-\mu\Delta\mathbf{u})$" "\n"
     r"$\rho_t=-\nabla\cdot(\rho\mathbf{u})$"),
]


def load(case: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "examples" / case / "results" / name)
    # Part I writes the mean absolute coefficient error under the name MAE;
    # the paper calls the same quantity L1.
    return df.replace({"metric": {"MAE": "L1"}})


def bubbles(ax, values_by_rung, *, log_x: bool, n_bins: int = 7) -> None:
    """One row per rung, binned over that rung's own range and scaled to its peak.

    Per-row bins because the rungs differ by orders of magnitude here: a shared
    binning puts every accurate rung in one column at the left and resolves only
    the worst one.
    """
    for row, rung in enumerate(RUNGS):
        v = np.asarray(values_by_rung[rung], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        y = len(RUNGS) - 1 - row
        if log_x:
            v = v[v > 0]
            if v.size == 0:
                continue
            lo, hi = v.min(), v.max()
            edges = (np.geomspace(lo, hi, n_bins + 1) if hi > lo
                     else np.array([lo * 0.9, lo * 1.1]))
        else:
            lo, hi = v.min(), v.max()
            edges = (np.linspace(lo, hi, n_bins + 1) if hi > lo
                     else np.array([lo - 1e-9, lo + 1e-9]))
        counts, _ = np.histogram(v, bins=edges)
        centres = np.sqrt(edges[:-1] * edges[1:]) if log_x else 0.5 * (edges[:-1] + edges[1:])
        keep = counts > 0
        if not keep.any():
            continue
        sizes = 480.0 * counts[keep] / counts[keep].max()
        ax.scatter(centres[keep], np.full(keep.sum(), y), s=sizes,
                   color=COLORS[rung], alpha=0.80, edgecolors="none", zorder=3)
        ax.scatter([np.median(v)], [y], marker="|", s=260,
                   color="black", linewidths=1.4, zorder=4)

    ax.set_yticks(range(len(RUNGS)))
    ax.set_yticklabels([LABELS[r] for r in reversed(RUNGS)], fontsize=13)
    ax.set_ylim(-0.7, len(RUNGS) - 0.3)
    if log_x:
        ax.set_xscale("log")
    ax.grid(axis="x", color="0.9", lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "serif", "mathtext.fontset": "cm"})

    n = len(CASES)
    fig = plt.figure(figsize=(1920 / 72.0, 1080 / 72.0))
    gs = fig.add_gridspec(
        n, 4, width_ratios=[0.20, 0.28, 0.28, 0.24],
        left=0.035, right=0.985, top=0.925, bottom=0.085, hspace=0.45, wspace=0.22,
    )

    headers = ["Equation", r"$L_1$ error (coefficients)", r"$L_0$ error (support)",
               "Medians"]
    for col, text in enumerate(headers):
        ax = fig.add_subplot(gs[0, col]); ax.axis("off")
        ax.set_title(text, fontsize=21, fontweight="bold", pad=26)

    for row, (title, case, name, equation) in enumerate(CASES):
        df = load(case, name)
        l1 = {r: df[(df.model == r) & (df.metric == "L1")]["value"].to_numpy()
              for r in RUNGS}
        l0 = {r: df[(df.model == r) & (df.metric == "L0")]["value"].to_numpy()
              for r in RUNGS}

        ax = fig.add_subplot(gs[row, 0]); ax.axis("off")
        ax.text(0.0, 0.62, title, fontsize=17, fontweight="bold", va="center")
        ax.text(0.0, 0.18, equation, fontsize=14, va="center")

        ax_l1 = fig.add_subplot(gs[row, 1])
        bubbles(ax_l1, l1, log_x=True)
        ax_l1.set_xlabel(r"$L_1$", fontsize=13, labelpad=1)

        ax_l0 = fig.add_subplot(gs[row, 2])
        bubbles(ax_l0, l0, log_x=False)
        ax_l0.set_xlabel(r"$L_0$", fontsize=13, labelpad=1)

        ax_t = fig.add_subplot(gs[row, 3]); ax_t.axis("off")
        med_l1 = {r: float(np.median(l1[r])) for r in RUNGS}
        med_l0 = {r: float(np.median(l0[r])) for r in RUNGS}
        best_l1 = min(med_l1, key=med_l1.get)
        ax_t.text(0.03, 0.95, "model", fontsize=12, fontweight="bold")
        ax_t.text(0.42, 0.95, r"$L_1$", fontsize=12, fontweight="bold")
        ax_t.text(0.74, 0.95, r"$L_0$", fontsize=12, fontweight="bold")
        ax_t.plot([0.02, 0.98], [0.90, 0.90], color="0.3", lw=1.0)
        for i, r in enumerate(RUNGS):
            y = 0.80 - i * 0.135
            bold = "bold" if r == best_l1 else "normal"
            ax_t.text(0.03, y, LABELS[r], fontsize=12, color=COLORS[r], fontweight="bold")
            ax_t.text(0.42, y, f"{med_l1[r]:.3g}", fontsize=12, fontweight=bold)
            ax_t.text(0.74, y, f"{med_l0[r]:.3g}", fontsize=12, fontweight=bold)
        ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

    n_runs = {c: int(load(c, nm).groupby(["model", "metric"]).size().iloc[0])
              for _, c, nm, _ in CASES}
    fig.text(0.035, 0.022,
             "Bubble area is the count in that rung's own bin; the vertical tick is the median. "
             f"Runs per rung: {', '.join(f'{k} {v}' for k, v in n_runs.items())}.",
             fontsize=12, color="0.35")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "figure2_regenerated.pdf"
    fig.savefig(out, format="pdf", bbox_inches=None)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=72)
    print(f"wrote {out}")
    for _, case, name, _ in CASES:
        df = load(case, name)
        med = df[df.metric == "L1"].groupby("model")["value"].median()
        print(f"  {case:<16} " + "  ".join(f"{LABELS[r]}={med[r]:.3g}" for r in RUNGS))


if __name__ == "__main__":
    main()
