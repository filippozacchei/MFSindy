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

- ``double_pendulum_forecast_polynomial``: the forecast of the four states over
  the forecast window, for FORECAST_RUNGS: each rung's ensemble forecast (the
  median of its bag rollouts, as the ``ens_*`` metrics score it) and the band of
  its bounded bags, after the last seconds of training.
- ``double_pendulum_forecast_derivatives``: the same rungs' right sides along the
  true future states, against the true derivatives.
- ``double_pendulum_forecast_estimated``: the forecast figure for WMF alone,
  whitened with the true variances and with variances estimated from the data,
  each with its band: the same run, on the same data.
- ``double_pendulum_noise_study``: the ensemble's unexplained variance 1 - R^2
  (log scale, so near-perfect forecasts stay apart) and valid time over the
  whole forecast, and the coverage of the band, against the LF noise level, for
  NOISE_STUDY_RUNGS with true (filled) and estimated (open) variances: medians
  over the Monte Carlo, with interquartile bars for true variances. Every configuration of
  the study with results is drawn; missing ones are left out.
- ``double_pendulum_noise_estimate``: for one HF and one LF sensor of the run,
  the noise on omega_2 with its true and its estimated standard deviation
  (mfsindy.noise_estimation), over the first NOISE_SHOWN seconds.

The run shown is the one whose trajectory has the median energy over the Monte
Carlo's runs (median_run): the most typical trajectory, chosen from the data
alone, before any method is looked at. Its forecasts, with true and with
estimated variances, are recomputed once and kept beside the results
(run_record); ``--record`` draws another record instead, e.g.
a draft run's. Every figure shows the same run.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle

from mfsindy.cases import double_pendulum_forecasting as D
from mfsindy.noise_estimation import estimate_noise_variance
from mfsindy.plots import DISPLAY, PAPER_RC, PAPER_WIDTH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

HF, LF, GREY = "#2f6fae", "#c0392b", "0.55"
STATES = [r"$\theta_1$", r"$\omega_1$", r"$\theta_2$", r"$\omega_2$"]
#: Seconds between the observations drawn, whatever the sampling step.
SHOW_EVERY = 0.02
#: Rungs whose forecasts are drawn: one fidelity alone, naive fusion, weighted
#: fusion, each as (colour, line style). The colours are Okabe-Ito, which stay
#: apart under the common colour-vision deficiencies, and the line styles keep
#: the three apart in greyscale too.
FORECAST_RUNGS = {"HF": ("#0072B2", (0, (1, 1))),
                  "MF": ("#E69F00", (0, (5, 1.5, 1, 1.5))),
                  "MF_w": ("#009E73", "-")}
#: The known-against-estimated figure: WMF in its colour, estimated in Okabe-Ito
#: vermillion, dashed.
KNOWN_ESTIMATED = (("known $\\sigma$", "#009E73", "-"), ("estimated $\\sigma$", "#D55E00", (0, (3, 1.5))))
#: The noise study: its LF levels, and the rungs drawn (in the forecast figure's
#: colours and styles, with VMF added as WMF's closest rival).
NOISE_LEVELS = (0.05, 0.1, 0.25, 0.5)
NOISE_STUDY_RUNGS = {**FORECAST_RUNGS, "VMF": ("#CC79A7", (0, (4, 1.5)))}
#: Seconds and state component the noise-estimate figure shows.
NOISE_SHOWN, NOISE_STATE = 4.0, 3
#: Seconds of training drawn before the forecast, with its data in greys.
LEAD = 2.0
#: Seconds of the forecast window the observations figure shows past training.
FORECAST_SHOWN = 1.0
#: The setup figures are subfigures printed at 0.49 of the text width.
WIDTH = 0.49 * PAPER_WIDTH
#: Seconds of the outer mass's path drawn behind the schematic.
TAIL = 0.35
#: The state the schematic shows, as (theta_1, omega_1, theta_2, omega_2).
SKETCH = np.array([0.33, 0.5, -0.87, -1.0])


def make_config(**overrides):
    """The config of ``examples/double_pendulum/main3.py``, as the Monte Carlo builds it."""

    path = ROOT / "examples" / "double_pendulum" / "main3.py"
    spec = importlib.util.spec_from_file_location("double_pendulum_main3", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module.make_config(**overrides)


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


def configuration(cfg, clean, noisy) -> None:
    """theta_2 against theta_1: the LF observations under the HF ones, around the trajectory."""

    k = max(1, int(round(SHOW_EVERY / cfg.dt)))
    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.9, bottom=0.15)
    for X in noisy.lf:
        ax.plot(X[::k, 0], X[::k, 2], ".", ms=0.5, color=LF, alpha=0.2, rasterized=True)
    ax.plot(noisy.hf[0][::k, 0], noisy.hf[0][::k, 2], ".", ms=0.8, color=HF, alpha=0.8, rasterized=True)
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
    k = max(1, int(round(SHOW_EVERY / cfg.dt)))
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_WIDTH, 0.5 * PAPER_WIDTH), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.11, hspace=0.15, wspace=0.22)
    for d, ax in enumerate(axes.ravel()):
        for X in noisy.lf:
            ax.plot(t[::k], X[::k, d], ".", ms=0.4, color=LF, alpha=0.2, rasterized=True)
        ax.plot(t[::k], noisy.hf[0][::k, d], ".", ms=0.6, color=HF, alpha=0.9, rasterized=True)
        ax.plot(t, clean[:, d], color="black", lw=0.5, ls=(0, (4, 2)))
        ax.plot(t_future, future[:, d], color="black", lw=0.5, ls=(0, (4, 2)))
        ax.axvspan(cfg.T_train, t_future[-1], color="0.93", lw=0, zorder=0)
        ax.set_xlim(t[0], cfg.T_train + FORECAST_SHOWN)
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


def energy(cfg, y: np.ndarray) -> float:
    """Mechanical energy of the double pendulum at state y = (theta_1, omega_1,
    theta_2, omega_2), point masses on massless rods, up to a constant."""

    t1, w1, t2, w2 = y
    m1, m2, l1, l2, g = cfg.m1, cfg.m2, cfg.l1, cfg.l2, cfg.g
    kinetic = 0.5 * (m1 + m2) * (l1 * w1) ** 2 + 0.5 * m2 * (l2 * w2) ** 2 \
        + m2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2)
    potential = -(m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2)
    return float(kinetic + potential)


def median_run(cfg) -> int:
    """The run whose trajectory has the median energy (the upper median for an
    even number of runs). Energy is conserved, so the starting state gives it."""

    starts = {k: D.sample_double_pendulum_initial_condition(
                  np.random.default_rng(cfg.seed_base + k), theta_range=cfg.theta_range,
                  omega_range=cfg.omega_range)
              for k in range(cfg.n_runs)}
    order = sorted(starts, key=lambda k: energy(cfg, starts[k]))
    return order[len(order) // 2]


def run_record(cfg, run: int) -> dict:
    """One run's record with its forecasts, computed once and kept beside the
    Monte Carlo's results (the Monte Carlo keeps the last run's only)."""

    path = D.double_pendulum_record_path(cfg).with_name(f"{Path(cfg.results_filename).stem}_run{run}_record.pkl")
    if not path.exists():
        print(f"computing run {run} of {cfg.results_filename} with its forecasts (about 30 min)")
        _, record = D.run_double_pendulum_forecast(cfg, run, keep_forecasts=True)
        with open(path, "wb") as fh:
            pickle.dump(record, fh)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_record(path: Path) -> dict:
    """The run record holding forecasts: the last of a Monte Carlo's records, or
    a single run's (``{"record": ...}``, as a draft run writes it)."""

    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict) and "record" in data:
        return data["record"]
    return data[-1]


def forecast_panels(cfg, record, key_truth, key_bags, labels, lead, name) -> None:
    """Four panels, one per state: truth dashed, each rung's ensemble median and
    band; with ``lead`` = (t, clean, observed), the end of training in greys."""

    rungs = record["rungs"]
    first = rungs[next(iter(FORECAST_RUNGS))]["forecast"]
    n = first[key_truth].shape[0]
    t_future = cfg.T_train + np.arange(n) * cfg.forecast_dt
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_WIDTH, 0.5 * PAPER_WIDTH), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.11, hspace=0.15, wspace=0.22)
    for d, ax in enumerate(axes.ravel()):
        if lead is not None:
            t, clean, observed = lead
            k = max(1, int(round(SHOW_EVERY / cfg.dt)))
            for X in observed.lf:
                ax.plot(t[::k], X[::k, d], ".", ms=0.4, color="0.75", rasterized=True)
            ax.plot(t[::k], observed.hf[0][::k, d], ".", ms=0.7, color="0.25", rasterized=True)
            ax.plot(t, clean[:, d], color="black", lw=0.6, ls=(0, (4, 2)))
            ax.axvline(cfg.T_train, color="0.5", lw=0.5, ls=":")
        for rung, (color, style) in FORECAST_RUNGS.items():
            bags = rungs[rung]["forecast"][key_bags]
            bounded = bags[np.all(np.isfinite(bags), axis=(1, 2))]
            if len(bounded):
                lower, upper = np.quantile(bounded, cfg.band, axis=0)
                ax.fill_between(t_future, lower[:, d], upper[:, d], color=color, alpha=0.15, lw=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)            # all-NaN times
                median = np.nanmedian(bags, axis=0)
            ax.plot(t_future, median[:, d], color=color, lw=1.0, ls=style)
        ax.plot(t_future, first[key_truth][:, d], color="black", lw=0.6, ls=(0, (4, 2)))
        ax.set_xlim((lead[0][0] if lead is not None else t_future[0]), t_future[-1])
        ax.set_ylabel(labels[d], labelpad=1)
        style_panel(ax)
    for ax in axes[1]:
        ax.set_xlabel("$t$", labelpad=1)
    handles = [Line2D([], [], color="black", lw=0.6, ls=(0, (4, 2)), label="reference")]
    if lead is not None:
        handles += [Line2D([], [], ls="none", marker=".", color="0.25", label="HF data"),
                    Line2D([], [], ls="none", marker=".", color="0.75", label="LF data")]
    handles += [Line2D([], [], color=c, lw=1.0, ls=st, label=f"{DISPLAY.get(r, r)} median")
                for r, (c, st) in FORECAST_RUNGS.items()]
    handles += [plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.25, lw=0,
                              label=f"{round(100 * (cfg.band[1] - cfg.band[0]))}% band")]
    fig.legend(handles=handles, loc="upper center", ncol=(len(handles) + 1) // 2, frameon=False,
               handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    save(fig, name)


def forecasts(cfg, record, clean, observed, t) -> None:
    keep = t >= cfg.T_train - LEAD
    lead_data = type(observed)(hf=[X[keep] for X in observed.hf], lf=[X[keep] for X in observed.lf],
                               t_argument=observed.t_argument)
    forecast_panels(cfg, record, "truth", "bags", STATES, (t[keep], clean[keep], lead_data),
                    "double_pendulum_forecast_polynomial")
    dots = [r"$\dot\theta_1$", r"$\dot\omega_1$", r"$\dot\theta_2$", r"$\dot\omega_2$"]
    forecast_panels(cfg, record, "true_rhs", "bag_rhs", dots, None,
                    "double_pendulum_forecast_derivatives")


def forecast_known_estimated(cfg, record, estimated, clean, observed, t) -> None:
    """WMF's forecast with known and with estimated variances, each with its band."""

    keep = t >= cfg.T_train - LEAD
    k = max(1, int(round(SHOW_EVERY / cfg.dt)))
    truth = record["rungs"]["MF_w"]["forecast"]["truth"]
    t_future = cfg.T_train + np.arange(truth.shape[0]) * cfg.forecast_dt
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_WIDTH, 0.5 * PAPER_WIDTH), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.11, hspace=0.15, wspace=0.22)
    for d, ax in enumerate(axes.ravel()):
        for X in observed.lf:
            ax.plot(t[keep][::k], X[keep][::k, d], ".", ms=0.4, color="0.75", rasterized=True)
        ax.plot(t[keep][::k], observed.hf[0][keep][::k, d], ".", ms=0.7, color="0.25", rasterized=True)
        ax.plot(t[keep], clean[keep, d], color="black", lw=0.6, ls=(0, (4, 2)))
        ax.axvline(cfg.T_train, color="0.5", lw=0.5, ls=":")
        for rec, (_, color, style) in zip((record, estimated), KNOWN_ESTIMATED):
            bags = rec["rungs"]["MF_w"]["forecast"]["bags"]
            bounded = bags[np.all(np.isfinite(bags), axis=(1, 2))]
            if len(bounded):
                lower, upper = np.quantile(bounded, cfg.band, axis=0)
                ax.fill_between(t_future, lower[:, d], upper[:, d], color=color, alpha=0.15, lw=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(bags, axis=0)
            ax.plot(t_future, median[:, d], color=color, lw=1.0, ls=style)
        ax.plot(t_future, truth[:, d], color="black", lw=0.6, ls=(0, (4, 2)))
        ax.set_xlim(t[keep][0], t_future[-1])
        ax.set_ylabel(STATES[d], labelpad=1)
        style_panel(ax)
    for ax in axes[1]:
        ax.set_xlabel("$t$", labelpad=1)
    handles = [Line2D([], [], color="black", lw=0.6, ls=(0, (4, 2)), label="reference"),
               Line2D([], [], ls="none", marker=".", color="0.25", label="HF data"),
               Line2D([], [], ls="none", marker=".", color="0.75", label="LF data")]
    handles += [Line2D([], [], color=c, lw=1.0, ls=st, label=f"WMF median, {name}") for name, c, st in KNOWN_ESTIMATED]
    handles += [plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.25, lw=0,
                              label=f"{round(100 * (cfg.band[1] - cfg.band[0]))}% band")]
    fig.legend(handles=handles, loc="upper center", ncol=(len(handles) + 1) // 2, frameon=False,
               handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    save(fig, "double_pendulum_forecast_estimated")


def noise_study(cfg) -> None:
    """Forecast scores against the LF noise level, with true and estimated variances."""

    import pandas as pd

    horizon = max(cfg.horizons)
    panels = ((f"ens_r2@{horizon:g}", rf"ensemble $1-R^2$, {horizon:g} s"),
              ("valid_time", "valid time (s)"),
              (f"state_coverage@{horizon:g}", rf"coverage, {horizon:g} s"))
    fig, axes = plt.subplots(1, 3, figsize=(PAPER_WIDTH, 0.34 * PAPER_WIDTH))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.2, wspace=0.32)
    for estimated in (False, True):
        tables = {}
        for level in NOISE_LEVELS:
            c = make_config(noise_lf_level=level, estimate_noise=estimated)
            path = Path(c.results_dir) / c.results_filename
            if path.exists():
                tables[level] = pd.read_csv(path)
        for ax, (metric, title) in zip(axes, panels):
            for rung, (color, style) in NOISE_STUDY_RUNGS.items():
                # HF and MF use no variances, so their estimated-variance curves
                # repeat the true ones and are not drawn twice.
                if estimated and rung in ("HF", "MF"):
                    continue
                values = {level: t.loc[(t.metric == metric) & (t.model == rung), "value"]
                          for level, t in tables.items()}
                if metric.startswith("ens_r2"):
                    values = {level: (1.0 - v).clip(lower=1e-4) for level, v in values.items()}
                if not values:
                    continue
                x = np.array(list(values))
                lo, mid, hi = np.array([v.quantile([0.25, 0.5, 0.75]).to_numpy() for v in values.values()]).T
                # A small offset per rung keeps the bars apart on the log axis.
                shift = 1.0 + 0.035 * (list(NOISE_STUDY_RUNGS).index(rung) - 1.5)
                if not estimated:
                    ax.errorbar(x * shift, mid, yerr=[mid - lo, hi - mid], color=color, lw=0, elinewidth=0.6,
                                capsize=0, alpha=0.6)
                ax.plot(x * shift, mid, color=color, ls=style, lw=1.0 if not estimated else 0.8,
                        alpha=1.0 if not estimated else 0.75, marker="o", ms=2.5,
                        mfc=color if not estimated else "white")
            ax.set_title(title, pad=2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(NOISE_LEVELS, [f"{100 * l:g}%" for l in NOISE_LEVELS])
        ax.minorticks_off()
        ax.set_xlabel("LF noise level $b$", labelpad=1)
        style_panel(ax)
    axes[0].set_yscale("log")
    axes[1].set_ylim(0, max(cfg.horizons))
    axes[2].set_ylim(0.5, 1.0)
    axes[2].axhline(cfg.band[1] - cfg.band[0], color="0.6", lw=0.6, ls=":")
    handles = [Line2D([], [], color=c, ls=st, lw=1.0, label=DISPLAY.get(r, r)) for r, (c, st) in NOISE_STUDY_RUNGS.items()]
    handles += [Line2D([], [], color="0.3", marker="o", ms=2.5, ls="none", label=r"true $\sigma$"),
                Line2D([], [], color="0.3", marker="o", ms=2.5, mfc="white", ls="none", label=r"estimated $\sigma$")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               columnspacing=1.0, handletextpad=0.3, borderaxespad=0.2)
    save(fig, "double_pendulum_noise_study")


def noise_estimate(cfg, clean, observed, t) -> None:
    """The noise on one state for an HF and an LF sensor, with its true and its
    estimated standard deviation."""

    d, keep = NOISE_STATE, t <= NOISE_SHOWN
    fig, axes = plt.subplots(2, 1, figsize=(PAPER_WIDTH, 0.4 * PAPER_WIDTH), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.13, hspace=0.2)
    for ax, name, x, level, color in ((axes[0], "HF", observed.hf[0], cfg.noise_hf_level, HF),
                                      (axes[1], "LF", observed.lf[0], cfg.noise_lf_level, LF)):
        true = np.sqrt(D.double_pendulum_noise_variance(clean, level))[:, d]
        est = np.sqrt(estimate_noise_variance(x))[:, d]
        ax.plot(t[keep], np.abs(x - clean)[keep, d], ".", ms=0.8, color=color, alpha=0.35, rasterized=True)
        ax.plot(t[keep], true[keep], color="black", lw=0.7, ls=(0, (4, 2)))
        ax.plot(t[keep], est[keep], color=color, lw=0.9)
        ax.set_ylabel(name, labelpad=1)
        ax.set_xlim(0, NOISE_SHOWN)
        style_panel(ax)
    axes[1].set_xlabel("$t$", labelpad=1)
    fig.legend(handles=[Line2D([], [], ls="none", marker=".", color="0.4", label=rf"$|\varepsilon|$ on {STATES[d]}"),
                        Line2D([], [], color="black", lw=0.7, ls=(0, (4, 2)), label=r"true $\sigma$"),
                        Line2D([], [], color="0.4", lw=0.9, label=r"estimated $\hat\sigma$")],
               loc="upper center", ncol=3, frameon=False, columnspacing=1.5, borderaxespad=0.2)
    save(fig, "double_pendulum_noise_estimate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, default=None,
                        help="run record with forecasts (default: the Monte Carlo's)")
    args = parser.parse_args()

    plt.rcParams.update(PAPER_RC)
    cfg = make_config()
    estimated = None
    if args.record is not None:
        record = load_record(args.record)
    elif (Path(cfg.results_dir) / cfg.results_filename).exists():
        record = run_record(cfg, median_run(cfg))
        cfg_estimated = make_config(estimate_noise=True)
        if (Path(cfg_estimated.results_dir) / cfg_estimated.results_filename).exists():
            estimated = run_record(cfg_estimated, record["run"])
    else:
        record = None
    run = cfg.n_runs - 1 if record is None else record["run"]
    clean, noisy, t, future = run_data(cfg, run)
    schematic(cfg)
    configuration(cfg, clean, noisy)
    observations(cfg, clean, noisy, t, future)
    noise_estimate(cfg, clean, noisy, t)
    noise_study(cfg)
    if record is None:
        print("no Monte Carlo results; forecast figures skipped")
        return
    forecasts(cfg, record, clean, noisy, t)
    if estimated is not None:
        forecast_known_estimated(cfg, record, estimated, clean, noisy, t)


if __name__ == "__main__":
    main()
