"""Accuracy against kappa, with the strategy held fixed: ``kappa_accuracy``.

Part I's Monte Carlo keeps every run's error at every candidate support
(``errors_by_support`` in the selection record), so each strategy's median L1
can be read along the supports, against kappa there -- not at the support each
strategy happened to choose, where the strategies differ for reasons of their
own. One panel per ODE case; kappa is the median over clean reference
trajectories, as each case's design report gives it. A hollow marker is a
support the strategy could not choose: its weak system has fewer than two
equations per library term.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mfsindy.cases import hopf, lorenz, pendulum
from mfsindy.experiments import load_selection
from mfsindy.experiments.selection import MIN_ROWS_PER_TERM
from mfsindy.plots import DISPLAY, PAPER_RC, PAPER_WIDTH, PART1_COLORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

CASES = (("lorenz", "Lorenz", lorenz.lorenz_kappa_by_support),
         ("hopf", "Hopf", hopf.hopf_kappa_by_support),
         ("pendulum", "Pendulum", pendulum.pendulum_kappa_by_support))
RUNGS = ("HF", "MF", "VMF", "MF_w")


def make_config(case: str):
    path = ROOT / "examples" / case / "main1.py"
    spec = importlib.util.spec_from_file_location(f"{case}_main1", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module.make_config()


def main() -> None:
    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(1, 3, figsize=(PAPER_WIDTH, 0.34 * PAPER_WIDTH))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.2, wspace=0.3)
    for ax, (case, title, kappa_by_support) in zip(axes, CASES):
        cfg = make_config(case)
        table = kappa_by_support(cfg, cfg.H_grid)
        record = load_selection(cfg)
        errors = record["errors_by_support"]
        n_terms = np.asarray(record["true"]).shape[0]                  # (library terms, equations)
        for rung in RUNGS:
            n_traj = cfg.n_hf if rung == "HF" else cfg.n_hf + cfg.n_lf
            x, mid, lo, hi, ok = [], [], [], [], []
            for H, kappa, K in zip(table["H_xt"], table["kappa_median"], table["K"]):
                e = errors[rung][float(H)][:, 0]
                x.append(kappa)
                mid.append(np.median(e)); lo.append(np.quantile(e, 0.25)); hi.append(np.quantile(e, 0.75))
                ok.append(n_traj * K >= MIN_ROWS_PER_TERM * n_terms)
            x, mid, lo, hi, ok = map(np.asarray, (x, mid, lo, hi, ok))
            color = PART1_COLORS[rung]
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(x, mid, color=color, lw=0.9)
            ax.plot(x[ok], mid[ok], "o", ms=3, color=color)
            ax.plot(x[~ok], mid[~ok], "o", ms=3, mfc="white", color=color)
        ax.axvline(1.0, color="0.6", lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, pad=2)
        ax.set_xlabel(r"$\kappa$", labelpad=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"median $L_1$", labelpad=1)
    handles = [Line2D([], [], color=PART1_COLORS[r], lw=0.9, marker="o", ms=3, label=DISPLAY.get(r, r)) for r in RUNGS]
    handles += [Line2D([], [], color="0.4", ls="none", marker="o", ms=3, mfc="white", label="not admissible")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, columnspacing=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "kappa_accuracy.pdf")
    fig.savefig(OUT / "kappa_accuracy.png", dpi=300)
    print(f"wrote {OUT / 'kappa_accuracy.pdf'}")


if __name__ == "__main__":
    main()
