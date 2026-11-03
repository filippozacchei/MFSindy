"""Accuracy against kappa, with the strategy held fixed: ``kappa_accuracy_part{1,2}``.

The Monte Carlo of each part keeps every run's error at every candidate support
(``errors_by_support`` in the selection record), so each strategy's median L1
can be read along the supports, against kappa there -- not at the support each
strategy happened to choose, where the strategies differ for reasons of their
own. One panel per case, kappa as each case's design report gives it: Part I
the median over clean reference trajectories, Part II on one noisy reference
with sigma carried. A hollow marker is a support the strategy could not choose:
its weak system has fewer than two equations per library term. A PDE has two
spatial supports, one line each. Every figure is drawn twice: median L1 with
its interquartile band, and the exact-support rate (share of runs with L0 = 0,
whose median is 0 almost everywhere), as ``*_exact``.

The kappa tables are cached in results/kappa (the PDE ones take minutes to
hours); a missing table is computed, and ``--tables`` recomputes them all. A
case whose record has no per-support errors yet is left as a blank panel.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mfsindy.cases import burgers, hopf, lorenz, pendulum, vorticity_flow
from mfsindy.experiments import load_selection
from mfsindy.experiments.selection import MIN_ROWS_PER_TERM
from mfsindy.plots import DISPLAY, PAPER_RC, PAPER_WIDTH, PART1_COLORS, PART2_COLORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "kappa"

CASES = (("lorenz", "Lorenz"), ("hopf", "Hopf"), ("pendulum", "Pendulum"),
         ("burgers", "Burgers"), ("isothermal_flow", "Isothermal flow"), ("vorticity", "Vorticity flow"))
PDES = ("burgers", "isothermal_flow", "vorticity")
RUNGS = {1: ("HF", "MF", "VMF", "MF_w"), 2: ("No weighting", "Ones GLS", "Variance diag", "Variance GLS")}
COLORS = {1: PART1_COLORS, 2: PART2_COLORS}
LABELS = {1: DISPLAY, 2: {"No weighting": "None", "Ones GLS": "Unit", "Variance diag": "Diag.", "Variance GLS": "Full"}}


def load_main(case: str, part: int):
    path = ROOT / "examples" / case / f"main{part}.py"
    spec = importlib.util.spec_from_file_location(f"{case}_main{part}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def compute_table(case: str, part: int, module, cfg, record) -> pd.DataFrame:
    """Kappa per support, as ``H_xt``, ``K`` (test functions on one field) and ``kappa_median``."""

    if part == 2:
        table = module.kappa_by_support(cfg, verbose=False)
        table["H_xt"] = [list(H) if np.ndim(H) else H for H in module.supports(cfg)]
        return table
    if case == "lorenz":
        return lorenz.lorenz_kappa_by_support(cfg, cfg.H_grid)
    if case == "hopf":
        return hopf.hopf_kappa_by_support(cfg, cfg.H_grid)
    if case == "pendulum":
        return pendulum.pendulum_kappa_by_support(cfg, cfg.H_grid)
    if case == "burgers":
        # kappa on whole flows, K on the window the runs observe
        table = burgers.burgers_kappa_by_support(cfg, cfg.H_grid)
        table["K"] = [burgers.burgers_weak_design(replace(cfg, H_xt=H, K=None))["K_used"] for H in cfg.H_grid]
        return table
    if case == "isothermal_flow":
        return module.design_report(cfg)
    return vorticity_flow.vorticity_kappa_by_support(cfg, cfg.H_grid, coefficients=np.asarray(record["true"]))


def kappa_table(case: str, part: int, module, cfg, record, *, refresh: bool) -> pd.DataFrame:
    path = TABLES / f"{case}_part{part}_kappa.pkl"
    if refresh or not path.exists():
        TABLES.mkdir(parents=True, exist_ok=True)
        compute_table(case, part, module, cfg, record).to_pickle(path)
    return pd.read_pickle(path)


def n_trajectories(part: int, rung: str, cfg) -> int:
    if part == 2:
        return cfg.n_traj
    return cfg.n_hf if rung == "HF" else cfg.n_hf + cfg.n_lf


METRICS = {"l1": (0, r"median $L_1$", ""), "exact": (1, r"exact support (%)", "_exact")}


def panel(ax, case: str, part: int, *, refresh: bool, metric: str = "l1") -> bool:
    module = load_main(case, part)
    cfg = module.make_config()
    record = load_selection(cfg)
    errors = record.get("errors_by_support")
    if errors is None:
        return False
    table = kappa_table(case, part, module, cfg, record, refresh=refresh)
    true = np.asarray(record["true"])                                   # ODEs (terms, equations), PDEs (equations, terms)
    n_terms = true.shape[1] if case in PDES else true.shape[0]
    for rung in RUNGS[part]:
        x, mid, lo, hi, ok, space = [], [], [], [], [], []
        for H, kappa, K in zip(table["H_xt"], table["kappa_median"], table["K"]):
            e = errors[rung][tuple(map(float, H)) if case in PDES else float(H)][:, METRICS[metric][0]]
            x.append(kappa)
            if metric == "l1":
                mid.append(np.median(e)); lo.append(np.quantile(e, 0.25)); hi.append(np.quantile(e, 0.75))
            else:
                mid.append(100 * np.mean(e == 0)); lo.append(np.nan); hi.append(np.nan)
            ok.append(n_trajectories(part, rung, cfg) * K >= MIN_ROWS_PER_TERM * n_terms)
            space.append(float(np.ravel(H)[0]) if case in PDES else 0.0)
        x, mid, lo, hi, ok, space = map(np.asarray, (x, mid, lo, hi, ok, space))
        color = COLORS[part][rung]
        # one line per spatial support, the narrower solid
        for ls, h in zip(("-", "--"), np.unique(space)):
            on = space == h
            order = np.argsort(x[on])
            if metric == "l1":
                ax.fill_between(x[on][order], lo[on][order], hi[on][order], color=color, alpha=0.1, lw=0)
            ax.plot(x[on][order], mid[on][order], color=color, lw=0.9, ls=ls)
        ax.plot(x[ok], mid[ok], "o", ms=3, color=color)
        ax.plot(x[~ok], mid[~ok], "o", ms=3, mfc="white", color=color)
    ax.axvline(1.0, color="0.6", lw=0.6, ls=":")
    ax.set_xscale("log")
    if metric == "l1":
        ax.set_yscale("log")
    else:
        ax.set_ylim(-5, 105)
    return True


def figure(part: int, *, refresh: bool, metric: str = "l1") -> None:
    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(2, 3, figsize=(PAPER_WIDTH, 0.62 * PAPER_WIDTH))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.1, wspace=0.3, hspace=0.5)
    for ax, (case, title) in zip(axes.flat, CASES):
        if not panel(ax, case, part, refresh=refresh, metric=metric):
            ax.text(0.5, 0.5, "pending", ha="center", va="center", color="0.5", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, pad=2)
        ax.set_xlabel(r"$\kappa$", labelpad=1)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(METRICS[metric][1], labelpad=1)
    rungs = RUNGS[part]
    handles = [Line2D([], [], color=COLORS[part][r], lw=0.9, marker="o", ms=3, label=LABELS[part].get(r, r))
               for r in rungs]
    handles += [Line2D([], [], color="0.4", ls="none", marker="o", ms=3, mfc="white", label="not admissible"),
                Line2D([], [], color="0.4", lw=0.9, label=r"$h_x = W/8$"),
                Line2D([], [], color="0.4", lw=0.9, ls="--", label=r"$h_x = W/4$")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, columnspacing=1.0,
               handlelength=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"kappa_accuracy_part{part}{METRICS[metric][2]}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    print(f"wrote {stem.with_suffix('.pdf')}")


def isothermal_figure(*, refresh: bool, metric: str = "l1") -> None:
    """The paper's version: the isothermal flow alone, Part I and Part II side by
    side; it is the one case that reaches far past kappa = 1."""

    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 0.42 * PAPER_WIDTH))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.27, wspace=0.2)
    for ax, part in zip(axes, (1, 2)):
        panel(ax, "isothermal_flow", part, refresh=refresh, metric=metric)
        ax.set_title(f"Part {'I' * part}", pad=2)
        ax.set_xlabel(r"$\kappa$", labelpad=1)
        ax.spines[["top", "right"]].set_visible(False)
        handles = [Line2D([], [], color=COLORS[part][r], lw=0.9, marker="o", ms=3, label=LABELS[part].get(r, r))
                   for r in RUNGS[part]]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(handles),
                  frameon=False, handlelength=1.6, columnspacing=1.0)
    axes[0].set_ylabel(METRICS[metric][1], labelpad=1)
    fig.legend(handles=[Line2D([], [], color="0.4", ls="none", marker="o", ms=3, mfc="white", label="not admissible"),
                        Line2D([], [], color="0.4", lw=0.9, label=r"$h_x = W/8$"),
                        Line2D([], [], color="0.4", lw=0.9, ls="--", label=r"$h_x = W/4$")],
               loc="upper center", ncol=3, frameon=False, handlelength=1.6)
    stem = OUT / f"kappa_accuracy_isothermal{METRICS[metric][2]}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    print(f"wrote {stem.with_suffix('.pdf')}")


def main() -> None:
    refresh = "--tables" in sys.argv
    for metric in METRICS:
        for part in (1, 2):
            figure(part, refresh=refresh and metric == "l1", metric=metric)
        isothermal_figure(refresh=False, metric=metric)


if __name__ == "__main__":
    main()
