"""The Part III data figures: the double pendulum and what one run observes.

Three figures, with the manuscript's file names. The last two are drawn through
the generators the Monte Carlo uses (examples/double_pendulum/main3.py), so they
show the data the identification sees.

- ``double_pendulum_schematic_tail``: the system, as forecasting.ipynb drew it --
  rods, masses, angles and angular velocities, over the outer mass's last TAIL
  seconds and a few ghosted earlier configurations. It shows an illustrative
  state (SKETCH) close to the one the notebook showed: the runs' own HF
  trajectories swing too little for the angles to read.
- ``double_pendulum_theta1_theta2``: configuration space, the run's trajectory
  and its HF and LF observations.
- ``double_pendulum_observations``: the four states against time, observed over the
  training window and continued over the forecast one.

The run shown is the last, whose forecasts the Monte Carlo keeps for the
forecast figure, so the two show the same run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle

from mfsindy.cases import double_pendulum_forecasting as D
from mfsindy.plots import PAPER_RC, PAPER_WIDTH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

HF, LF, GREY = "#2f6fae", "#c0392b", "0.55"
STATES = [r"$\theta_1$", r"$\omega_1$", r"$\theta_2$", r"$\omega_2$"]
#: The setup figures are subfigures printed at 0.49 of the text width.
WIDTH = 0.49 * PAPER_WIDTH
#: Seconds of the outer mass's path drawn behind the schematic.
TAIL = 0.35
#: The state the schematic shows, as (theta_1, omega_1, theta_2, omega_2).
SKETCH = np.array([0.33, 0.5, -0.87, -1.0])


def make_config():
    """The config of ``examples/double_pendulum/main3.py``, as the Monte Carlo builds it."""

    path = ROOT / "examples" / "double_pendulum" / "main3.py"
    spec = importlib.util.spec_from_file_location("double_pendulum_main3", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module.make_config()


def run_data(cfg, run: int):
    """The run's clean trajectory and observations, its time grid, and the true forecast."""

    t, clean = D.double_pendulum_clean_trajectory(cfg, run)
    observed = D.double_pendulum_forecast_batch(run, cfg)
    _, future = D.double_pendulum_forecast_case(cfg, run)
    return clean, observed, t, future


def style_panel(a) -> None:
    a.spines[["top", "right"]].set_visible(False)


def save(fig, name: str) -> None:
    # Uncropped, so the file is exactly the printed size.
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out}")


def cartesian(theta1, theta2, l1=1.0, l2=1.0):
    x1, y1 = l1 * np.sin(theta1), -l1 * np.cos(theta1)
    return x1, y1, x1 + l2 * np.sin(theta2), y1 - l2 * np.cos(theta2)


def sketch_path(cfg) -> np.ndarray:
    """The TAIL seconds leading to SKETCH. The pendulum is reversible, so they are
    the path from SKETCH with the velocities reversed, run backwards."""

    flip = np.array([1.0, -1.0, 1.0, -1.0])
    _, X = D.simulate_double_pendulum_trajectory(
        y0=SKETCH * flip, t_final=TAIL, dt=cfg.dt, m1=cfg.m1, m2=cfg.m2, l1=cfg.l1, l2=cfg.l2, g=cfg.g)
    return (X * flip)[::-1]


def curved_arrow(ax, xy, xytext, text, rad) -> None:
    ax.annotate(text, xy=xy, xytext=xytext, ha="center", va="center", zorder=8,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25", shrinkA=2, shrinkB=2,
                                connectionstyle=f"arc3,rad={rad}"))


def schematic(cfg) -> None:
    """Rods, masses, angles and angular velocities at SKETCH, over its recent path."""

    th1, om1, th2, om2 = sketch_path(cfg).T
    x1, y1, x2, y2 = cartesian(th1, th2, cfg.l1, cfg.l2)
    X1, Y1, X2, Y2 = x1[-1], y1[-1], x2[-1], y2[-1]

    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH))
    fig.subplots_adjust(0, 0, 1, 1)
    # The outer mass's path in the HF colour, fading in towards the state shown,
    # over ghosts of the earlier configurations.
    points = np.column_stack([x2, y2])
    fade = matplotlib.colors.to_rgba_array([HF] * (len(points) - 1))
    fade[:, 3] = np.linspace(0.1, 1.0, len(points) - 1)
    ax.add_collection(LineCollection(np.stack([points[:-1], points[1:]], axis=1), colors=fade,
                                     linewidths=1.2, capstyle="round", zorder=1))
    for k, alpha in zip(np.linspace(0, len(x1) - 2, 5, dtype=int), np.linspace(0.08, 0.3, 5)):
        ax.plot([0, x1[k], x2[k]], [0, y1[k], y2[k]], color=HF, lw=0.6, alpha=alpha, zorder=1)
        ax.add_patch(Circle((x2[k], y2[k]), 0.05, color=HF, alpha=alpha, lw=0, zorder=1))

    ax.plot([0, 0], [0, -1.2], ls="--", lw=0.5, color="0.8", zorder=2)
    ax.plot([X1, X1], [Y1, Y1 - 1.0], ls="--", lw=0.5, color="0.8", zorder=2)
    ax.plot([0, X1, X2], [0, Y1, Y2], color="black", lw=1.6, solid_capstyle="round", zorder=4)
    for (x, y), r in (((0, 0), 0.055), ((X1, Y1), 0.08), ((X2, Y2), 0.095)):
        ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="black", lw=0.9, zorder=7))

    # Each angle from the downward vertical, at the pivot and at the inner mass.
    for (cx, cy), theta, radius, label in (((0, 0), th1[-1], 0.34, STATES[0]),
                                           ((X1, Y1), th2[-1], 0.24, STATES[2])):
        phi = np.degrees(theta) - 90.0
        ax.add_patch(Arc((cx, cy), 2 * radius, 2 * radius, theta1=min(-90, phi), theta2=max(-90, phi),
                         lw=0.8, color="0.25", zorder=6))
        mid = np.radians((-90 + phi) / 2)
        ax.text(cx + 1.5 * radius * np.cos(mid), cy + 1.5 * radius * np.sin(mid), label,
                ha="center", va="center", zorder=8)

    # Angular velocities: an arrow onto each rod, from the side it turns towards.
    u1 = np.array([X1, Y1]) / np.hypot(X1, Y1)
    curved_arrow(ax, 0.22 * u1, (-0.48, -0.18), STATES[1], -0.35 if om1[-1] >= 0 else 0.35)
    u2 = np.array([X2 - X1, Y2 - Y1]) / np.hypot(X2 - X1, Y2 - Y1)
    curved_arrow(ax, np.array([X1 - 0.1, Y1]) + 0.2 * u2, (X1 + 0.1, Y1 + 0.2), STATES[3],
                 -0.35 if om2[-1] >= 0 else 0.35)

    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-2.12, 0.28)
    ax.axis("off")
    save(fig, "double_pendulum_schematic_tail")


def configuration(clean, noisy) -> None:
    """theta_2 against theta_1: the LF observations under the HF ones, around the trajectory."""

    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.9, bottom=0.15)
    for X in noisy.lf:
        ax.plot(X[::20, 0], X[::20, 2], ".", ms=0.5, color=LF, alpha=0.2, rasterized=True)
    ax.plot(noisy.hf[0][::20, 0], noisy.hf[0][::20, 2], ".", ms=0.8, color=HF, alpha=0.8, rasterized=True)
    ax.plot(clean[:, 0], clean[:, 2], color="black", lw=0.5, ls=(0, (4, 2)))
    ax.set_xlabel(STATES[0], labelpad=1)
    ax.set_ylabel(STATES[2], labelpad=1)
    style_panel(ax)
    handles = [Line2D([], [], ls="none", marker=".", color=HF, label="HF"),
               Line2D([], [], ls="none", marker=".", color=LF, label="LF"),
               Line2D([], [], color="black", lw=0.5, ls=(0, (4, 2)), label="reference")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, handletextpad=0.2,
               columnspacing=1.0, borderaxespad=0.2)
    save(fig, "double_pendulum_theta1_theta2")


def observations(cfg, clean, noisy, t, future) -> None:
    """The four states against time; the forecast window shaded after training."""

    t_future = cfg.T_train + np.arange(future.shape[0]) * cfg.forecast_dt
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_WIDTH, 0.5 * PAPER_WIDTH), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.11, hspace=0.15, wspace=0.22)
    for d, ax in enumerate(axes.ravel()):
        for X in noisy.lf:
            ax.plot(t[::20], X[::20, d], ".", ms=0.4, color=LF, alpha=0.2, rasterized=True)
        ax.plot(t[::20], noisy.hf[0][::20, d], ".", ms=0.6, color=HF, alpha=0.9, rasterized=True)
        ax.plot(t, clean[:, d], color="black", lw=0.5, ls=(0, (4, 2)))
        ax.plot(t_future, future[:, d], color="black", lw=0.5, ls=(0, (4, 2)))
        ax.axvspan(cfg.T_train, t_future[-1], color="0.93", lw=0, zorder=0)
        ax.set_xlim(t[0], t_future[-1])
        ax.set_ylabel(STATES[d], labelpad=1)
        style_panel(ax)
    for ax in axes[1]:
        ax.set_xlabel("$t$", labelpad=1)
    handles = [Line2D([], [], ls="none", marker=".", color=HF, label="HF"),
               Line2D([], [], ls="none", marker=".", color=LF, label="LF"),
               Line2D([], [], color="black", lw=0.5, ls=(0, (4, 2)), label="reference"),
               plt.Rectangle((0, 0), 1, 1, color="0.93", label="forecast window")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0.2)
    save(fig, "double_pendulum_observations")


def main() -> None:
    plt.rcParams.update(PAPER_RC)
    cfg = make_config()
    run = cfg.n_runs - 1
    clean, noisy, t, future = run_data(cfg, run)
    schematic(cfg)
    configuration(clean, noisy)
    observations(cfg, clean, noisy, t, future)


if __name__ == "__main__":
    main()
