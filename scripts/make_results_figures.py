"""The Part I and Part II results figures, as tables drawn from the error tables.

One row per system: a thumbnail of the data of one run on a grey reference, the
governing equation, the coefficient error of every run as a violin per strategy
on a log scale with the median marked, the share of runs that recover the exact
support, and the medians in figures. Part I thumbnails show HF in blue and LF in
red; Part II thumbnails colour each sample by its noise level. The thumbnails are
drawn through the generators the Monte Carlo uses, as in make_data_figure.py.

The figures are drawn at the text width and saved uncropped, so they print at
their own size.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mfsindy.cases import hopf as H
from mfsindy.cases import isothermal_flow as N
from mfsindy.cases import pendulum as P
from mfsindy.cases import vorticity_flow as V
from mfsindy.plots import DISPLAY, PAPER_BOLD, PAPER_RC, PAPER_WIDTH, PART1_COLORS, PART2_COLORS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
HF, LF, GREY = "#2f6fae", "#c0392b", "0.55"

BOLD = PAPER_BOLD
#: Equations and table entries at the tick-label size, to fit the text width.
SMALL = PAPER_RC["xtick.labelsize"]

SYSTEMS = [
    ("Lorenz", "lorenz", "lorenz",
     r"$\dot{x}=\sigma(y-x)$" "\n" r"$\dot{y}=x(\rho-z)-y$" "\n" r"$\dot{z}=xy-\beta z$"),
    ("Hopf", "hopf", "hopf",
     r"$\dot{r}=\mu r-r^{3}$" "\n" r"$\dot{\theta}=\omega$"),
    ("Pendulum", "pendulum", "pendulum",
     r"$\dot{\theta}=\omega$" "\n" r"$\dot{\omega}=c\,\omega-\frac{g}{\ell}\,\theta$"),
    ("Burgers", "burgers", "burgers",
     r"$u_t=-u\,u_x+\nu\,u_{xx}$"),
    ("Isothermal flow", "isothermal_flow", "navierstokes",
     r"$\mathbf{u}_t=-(\mathbf{u}\cdot\nabla)\mathbf{u}$" "\n"
     r"$\quad-\rho^{-1}(\nabla p-\mu\Delta\mathbf{u})$" "\n"
     r"$\rho_t=-\nabla\cdot(\rho\mathbf{u})$"),
    ("Vorticity flow", "vorticity", "vorticity",
     r"$\omega_t=-(u\,\omega)_x-(v\,\omega)_y$" "\n"
     r"$\quad+\nu\,(\omega_{xx}+\omega_{yy})$"),
]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_main(case: str, part: int):
    """``examples/<case>/main<part>.py``, whose config and data the Monte Carlo uses."""

    path = ROOT / "examples" / case / f"main{part}.py"
    spec = importlib.util.spec_from_file_location(f"{case}_main{part}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def load(case: str, prefix: str, part: int) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "examples" / case / "results" / f"{prefix}_part{part}_errors.csv")
    # Part I writes the coefficient error as MAE and names its strategies "model";
    # Part II calls the same quantity L1 and its weightings "method".
    return df.rename(columns={"method": "model"}).replace({"metric": {"MAE": "L1"}})


def strip_3d(ax) -> None:
    ax.grid(False)
    ax.set_axis_off()


# --- Part I thumbnails: one run's data, HF in blue and LF in red on a grey reference

def ode_batch(module, prefix: str, case: str):
    cfg = load_main(case, 1).make_config()
    std = getattr(module, f"_{prefix}_reference_state_std")(cfg)
    batch = getattr(module, f"_{prefix}_batch")
    return batch(0, cfg, cfg.noise_hf_rel * std, cfg.noise_lf_rel * std), batch(0, cfg, 0.0, 0.0)


def thumb_lorenz(ax) -> None:
    attractor, trajectories, _ = load_script("make_data_figure").lorenz_data()
    ax.plot(*attractor.T, color=GREY, lw=0.2, alpha=0.3)
    for _, noisy, color, _ in trajectories:
        ax.plot(*noisy[::10].T, ".", ms=0.6, color=color, alpha=0.7, rasterized=True)
    strip_3d(ax)


def thumb_plane(ax, module, prefix: str, case: str) -> None:
    noisy, clean = ode_batch(module, prefix, case)
    for X in clean.hf + clean.lf:
        ax.plot(*np.asarray(X)[:, :2].T, color=GREY, lw=0.4, alpha=0.6)
    for data, color in ((noisy.lf, LF), (noisy.hf, HF)):
        for X in data:
            ax.plot(*np.asarray(X)[::10, :2].T, ".", ms=0.6, color=color, alpha=0.6, rasterized=True)
    ax.set_axis_off()


def thumb_burgers(ax) -> None:
    data_figure = load_script("make_data_figure")
    cfg, _, batch, windows = data_figure.burgers_data()
    U, x = batch.metadata["flow"], batch.metadata["x_flow"]
    t = np.arange(U.shape[1]) * cfg.dt
    T_, X_ = np.meshgrid(t[::20], x, indexing="xy")
    faces = np.full(X_.shape + (4,), matplotlib.colors.to_rgba(GREY, 0.25))
    for w, color in windows[::-1]:
        faces[np.asarray(batch.metadata["windows"][w])] = matplotlib.colors.to_rgba(color, 0.25)
    ax.plot_surface(X_, T_, U[:, ::20], facecolors=faces, linewidth=0, shade=False,
                    rstride=1, cstride=1, rasterized=True)
    # Each window's noisy samples on its strip.
    data = batch.hf + batch.lf
    for w, color in windows[::-1]:
        idx = np.asarray(batch.metadata["windows"][w])
        Tw, Xw = np.meshgrid(t[::100], x[idx][::4], indexing="xy")
        ax.plot(Xw.ravel(), Tw.ravel(), np.asarray(data[w])[::4, ::100, 0].ravel(), ".", ms=0.8,
                color=color, alpha=0.8, rasterized=True)
    ax.view_init(elev=28, azim=-50)
    strip_3d(ax)


def thumb_ns(ax) -> None:
    """The density at the final time as a grey surface, the HF and two LF windows
    tinted, and their noisy density samples on it."""

    cfg = load_main("isothermal_flow", 1).make_config()
    U, t, grid = N.generate_isothermal_ns_dataset(N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu,
                                                  RT=cfg.RT, seed=cfg.seed_base)
    # As the Monte Carlo does: noise relative to the density's spread on the reference flow.
    std = float(np.std(U[:, :, :, 2]))
    batch = N._ns_dataset_batch(0, cfg, cfg.noise_hf_rel * std, cfg.noise_lf_rel * std, grid=grid, t=t)
    shown = [(k, HF if k < cfg.n_hf else LF) for k in range(cfg.n_hf + 2)]
    flow_surface(ax, cfg, U, batch.metadata["windows"], batch.hf + batch.lf, shown)


def thumb_vorticity(ax) -> None:
    """The vorticity at the final time as a grey surface, the HF and two LF
    windows tinted, and their noisy vorticity samples on it."""

    cfg = load_main("vorticity", 1).make_config()
    U, t, grid = V.generate_vorticity_dataset(N=cfg.N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, nu=cfg.nu,
                                              ic=cfg.ic, seed=cfg.seed_base)
    # As the Monte Carlo does: noise relative to the vorticity's spread on the reference flow.
    spreads = np.std(U, axis=(0, 1, 2))
    std = float(spreads[2])
    batch = V._dataset_batch(0, cfg, cfg.noise_hf_rel * std, cfg.noise_lf_rel * std, grid=grid, t=t,
                             component_scale=spreads / std)
    shown = [(k, HF if k < cfg.n_hf else LF) for k in range(cfg.n_hf + 2)]
    flow_surface(ax, cfg, U, batch.metadata["windows"], batch.hf + batch.lf, shown)


def flow_surface(ax, cfg, U, windows, data, shown) -> None:
    """The density at the final time in grey, the shown windows tinted, and their
    samples on it; ``shown`` pairs each window with a colour, or with an array of
    colours, one per sample."""

    x = np.linspace(0.0, cfg.L, cfg.N, endpoint=False)
    X_, Y_ = np.meshgrid(x, x, indexing="ij")
    faces = np.full(X_.shape + (4,), matplotlib.colors.to_rgba(GREY, 0.25))
    tint = lambda c: matplotlib.colors.to_rgba(c if isinstance(c, str) else GREY, 0.25)
    for k, color in shown[::-1]:
        # Index arrays in window order (a window may wrap around the period).
        ix, iy = (np.asarray(a).ravel() for a in windows[k])
        faces[np.ix_(ix, iy)] = tint(color)
    ax.plot_surface(X_, Y_, U[:, :, -1, 2], facecolors=faces, linewidth=0, shade=False,
                    rstride=2, cstride=2, rasterized=True)
    for k, color in shown[::-1]:
        ix, iy = (np.asarray(a).ravel() for a in windows[k])
        Yw, Xw = np.meshgrid(x[iy][::4], x[ix][::4])
        z = np.asarray(data[k])[::4, ::4, -1, 2].ravel()
        colors = color if isinstance(color, str) else color[::4, ::4].reshape(-1, 4)
        ax.scatter(Xw.ravel(), Yw.ravel(), z, s=0.8, c=colors if not isinstance(colors, str) else None,
                   color=colors if isinstance(colors, str) else None, alpha=0.8, linewidths=0,
                   depthshade=False, rasterized=True)
    ax.view_init(elev=35, azim=-60)
    strip_3d(ax)


# --- Part II thumbnails: one run's data on a grey reference, each sample coloured
# by its noise level (light to dark), which is what the weighting responds to.

def noise_colors(sigma: np.ndarray, top: float) -> np.ndarray:
    return plt.get_cmap("Reds")(0.3 + 0.7 * np.clip(sigma / top, 0.0, 1.0))


def ode_part2(case: str):
    main = load_main(case, 2)
    t, clean, noisy, variance = main.run_trajectories(main.make_config(), 0)
    sigma = [np.sqrt(np.mean(np.asarray(v), axis=1)) for v in variance]
    return clean, noisy, sigma, max(s.max() for s in sigma)


def thumb2_ode(ax, case: str, dims: int) -> None:
    clean, noisy, sigma, top = ode_part2(case)
    for X in clean:
        ax.plot(*np.asarray(X)[:, :dims].T, color=GREY, lw=0.3, alpha=0.5)
    for X, s in list(zip(noisy, sigma))[:3]:
        X = np.asarray(X)[::10, :dims]
        ax.scatter(*X.T, s=0.6, c=noise_colors(s[::10], top), linewidths=0, rasterized=True,
                   **({"depthshade": False} if dims == 3 else {}))
    if dims == 3:
        strip_3d(ax)
    else:
        ax.set_axis_off()


def thumb2_burgers(ax) -> None:
    main = load_main("burgers", 2)
    cfg, observed = main.make_config(), {}
    _, t, _, noisy, variance = main.run_trajectories(cfg, 0, observed=observed)
    U, x = observed["flow"], observed["x_flow"]
    T_, X_ = np.meshgrid(t[::20], x, indexing="xy")
    ax.plot_surface(X_, T_, U[:, ::20], color=GREY, alpha=0.25, linewidth=0, shade=False, rasterized=True)
    top = max(np.sqrt(v).max() for v in variance)
    for k in range(3):
        idx = np.asarray(observed["windows"][k])
        Tw, Xw = np.meshgrid(t[::100], x[idx][::4], indexing="xy")
        sigma = np.sqrt(np.asarray(variance[k]))[::4, ::100]
        ax.scatter(Xw.ravel(), Tw.ravel(), np.asarray(noisy[k])[::4, ::100, 0].ravel(), s=0.8,
                   c=noise_colors(sigma.ravel(), top), linewidths=0, depthshade=False, rasterized=True)
    ax.view_init(elev=28, azim=-50)
    strip_3d(ax)


def thumb2_ns(ax, case: str = "isothermal_flow") -> None:
    main = load_main(case, 2)
    cfg, observed = main.make_config(), {}
    _, _, _, noisy, variance = main.run_trajectories(cfg, 0, observed=observed)
    # The density's noise level at the final time, which the surface shows.
    sigma = [np.sqrt(np.asarray(v)[:, :, -1, 2]) for v in variance[:3]]
    top = max(s.max() for s in sigma)
    shown = [(k, noise_colors(sigma[k], top)) for k in range(3)]
    flow_surface(ax, cfg, observed["flow"], observed["windows"], noisy, shown)


# --- the table

@dataclass
class Part:
    number: int
    strategies: list
    colors: dict
    names: dict
    thumbs: list
    height: float


def rows(part: Part) -> tuple:
    """Vertical extent shared by the violins, the bars and the median table: one
    row per strategy, and one more on top for the table's header."""

    return len(part.strategies) - 0.5, -1.5


def violins(ax, df: pd.DataFrame, part: Part) -> None:
    for row, s in enumerate(part.strategies):
        l1 = df[(df.model == s) & (df.metric == "L1")].value.to_numpy()
        l1 = np.log10(l1[l1 > 0])
        if l1.size > 1:
            body = ax.violinplot(l1, positions=[row], widths=0.65, vert=False, showextrema=False)["bodies"][0]
            body.set_facecolor(matplotlib.colors.to_rgba(part.colors[s], 0.35))
            body.set_edgecolor(part.colors[s]); body.set_linewidth(0.6)
        ax.plot([np.median(l1)] * 2, [row - 0.3, row + 0.3], color=part.colors[s], lw=1.4)
    ax.set_ylim(*rows(part))
    ax.set_yticks(range(len(part.strategies)), [part.names.get(s, s) for s in part.strategies])
    for tick, s in zip(ax.get_yticklabels(), part.strategies):
        tick.set_color(part.colors[s])
    ax.grid(axis="x", color="0.88", lw=0.5)
    ax.set_axisbelow(True)
    # Whole decades only: the violins are drawn on log10 of the error, and an axis
    # spanning less than a decade is widened to the decades around it.
    lo, hi = ax.get_xlim()
    if np.floor(hi) <= np.ceil(lo):
        ax.set_xlim(np.floor(lo), np.ceil(hi))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$"))


def exact_bars(ax, df: pd.DataFrame, part: Part) -> None:
    for row, s in enumerate(part.strategies):
        l0 = df[(df.model == s) & (df.metric == "L0")].value.to_numpy()
        ax.barh(row, 100 * np.mean(l0 == 0), height=0.6, color=part.colors[s])
    ax.set_ylim(*rows(part))
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.grid(axis="x", color="0.88", lw=0.5)
    ax.set_axisbelow(True)


def medians(ax, df: pd.DataFrame, part: Part) -> None:
    """A small ruled table per system: strategy, median L1 and exact-support share,
    the best of each in bold; its rows line up with the violins and bars."""

    l1 = {s: np.median(df[(df.model == s) & (df.metric == "L1")].value) for s in part.strategies}
    exact = {s: 100 * np.mean(df[(df.model == s) & (df.metric == "L0")].value == 0)
             for s in part.strategies}
    best_l1, best_exact = min(l1.values()), max(exact.values())
    columns = (0.14, 0.52, 0.86)
    for x, text in zip(columns, ("Model", "$L_1$", "Exact")):
        ax.text(x, -1, text, ha="center", va="center", fontdict={**BOLD, "size": SMALL})
    for row, s in enumerate(part.strategies):
        ax.text(columns[0], row, part.names.get(s, s), ha="center", va="center", color=part.colors[s],
                fontsize=SMALL)
        mantissa, exponent = f"{l1[s]:.2e}".split("e")
        if l1[s] == best_l1:
            ax.text(columns[1], row, rf"$\mathbf{{{mantissa}}}\cdot\mathbf{{10}}^{{\mathbf{{{int(exponent)}}}}}$",
                    ha="center", va="center", fontsize=SMALL, math_fontfamily="custom")
        else:
            ax.text(columns[1], row, rf"${mantissa}\cdot 10^{{{int(exponent)}}}$", ha="center", va="center",
                    fontsize=SMALL)
        ax.text(columns[2], row, f"{exact[s]:.0f}", ha="center", va="center",
                fontdict={**BOLD, "size": SMALL} if exact[s] == best_exact else {"size": SMALL})
    bottom, top = rows(part)
    for y, lw in ((top, 0.9), (-0.5, 0.5), (bottom, 0.9)):
        ax.plot([0, 1], [y, y], color="k", lw=lw)
    # Closed on both sides, with the two inner separators lighter.
    for x, lw in ((0.0, 0.9), (0.28, 0.4), (0.75, 0.4), (1.0, 0.9)):
        ax.plot([x, x], [top, bottom], color="k", lw=lw, solid_capstyle="butt")
    ax.set_ylim(*rows(part))
    ax.set_xlim(0, 1)
    ax.set_axis_off()


def rule(fig, y: float, lw: float) -> None:
    fig.add_artist(matplotlib.lines.Line2D([0.005, 0.995], [y, y], transform=fig.transFigure,
                                           color="k", lw=lw))


def figure(part: Part) -> None:
    # A table: a header band, then one band per system, each with the same inner
    # margins, so every row has the same height and the columns line up.
    # part.height is for five systems; each further system adds one band.
    fig = plt.figure(figsize=(PAPER_WIDTH, part.height * (len(SYSTEMS) + 0.25) / 5.25))
    outer = fig.add_gridspec(len(SYSTEMS) + 1, 1, height_ratios=[0.25] + [1] * len(SYSTEMS),
                             hspace=0.0, left=0.005, right=0.995, top=0.985, bottom=0.03)
    widths = [0.95, 1.4, 1.45, 0.85, 1.75]
    header = outer[0].subgridspec(1, 5, width_ratios=widths, wspace=0.18)
    for k, text in enumerate(["Test case", "Equation", "$L_1$ error", "Exact support (%)",
                              "Median values"]):
        a = fig.add_subplot(header[k])
        a.text(0.5, 0.4, text, ha="center", va="center", fontdict=BOLD)
        a.set_axis_off()
    for row, ((title, case, prefix, equation), (thumb, three_d)) in enumerate(zip(SYSTEMS, part.thumbs)):
        band = outer[row + 1].subgridspec(3, 5, width_ratios=widths, height_ratios=[0.08, 1, 0.3],
                                          wspace=0.18, hspace=0.0)
        ax = fig.add_subplot(band[:, 0], projection="3d" if three_d else None)
        thumb(ax)
        if three_d:
            ax.set_box_aspect(None, zoom=1.35)
        else:
            ax.set_aspect("equal", "datalim")
        # The system's name across the top of its row, not along the side of the
        # thumbnail, which leaves the width to the plots; anchored to the row so
        # every name sits at the same height.
        cell = band[0, 0].get_position(fig)
        fig.text((cell.x0 + cell.x1) / 2, outer[row + 1].get_position(fig).y1 - 0.006, title,
                 ha="center", va="top", fontsize=SMALL)
        eq = fig.add_subplot(band[1, 1])
        eq.text(0.0, 0.5, equation, va="center", ha="left", fontsize=SMALL)
        eq.set_axis_off()
        df = load(case, prefix, part.number)
        violins(fig.add_subplot(band[1, 2]), df, part)
        exact_bars(fig.add_subplot(band[1, 3]), df, part)
        medians(fig.add_subplot(band[1, 4]), df, part)
        if row < len(SYSTEMS) - 1:
            rule(fig, outer[row + 1].get_position(fig).y0, 0.4)
    rule(fig, outer[0].get_position(fig).y1, 1.0)
    rule(fig, outer[0].get_position(fig).y0, 0.6)
    rule(fig, outer[-1].get_position(fig).y0, 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"part{part.number}_results.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out}")


#: Part I in the order the argument runs: single fidelity, naive pooling, the two
#: simpler weightings Referee 2 asked for, then the proposed one.
PART1 = Part(1, ["HF", "LF", "MF", "PMF", "VMF", "MF_w"], PART1_COLORS, DISPLAY,
             [(thumb_lorenz, True), (lambda a: thumb_plane(a, H, "hopf", "hopf"), False),
              (lambda a: thumb_plane(a, P, "pendulum", "pendulum"), False),
              (thumb_burgers, True), (thumb_ns, True), (thumb_vorticity, True)], 6.6)
#: Part II from no weighting to the full heteroscedastic covariance; the short
#: names are the caption's: unweighted, unit-variance, diagonal and full GLS.
PART2 = Part(2, ["No weighting", "Ones GLS", "Variance diag", "Variance GLS"], PART2_COLORS,
             {"No weighting": "None", "Ones GLS": "Unit", "Variance diag": "Diag.", "Variance GLS": "Full"},
             [(lambda a: thumb2_ode(a, "lorenz", 3), True), (lambda a: thumb2_ode(a, "hopf", 2), False),
              (lambda a: thumb2_ode(a, "pendulum", 2), False), (thumb2_burgers, True), (thumb2_ns, True),
              (lambda a: thumb2_ns(a, "vorticity"), True)],
             5.2)


def main() -> None:
    with plt.rc_context(PAPER_RC):
        for part in (PART1, PART2):
            missing = [case for _, case, prefix, _ in SYSTEMS
                       if not (ROOT / "examples" / case / "results" / f"{prefix}_part{part.number}_errors.csv").exists()]
            if missing:
                print(f"Part {part.number} skipped: no error table yet for {', '.join(missing)}")
                continue
            figure(part)


if __name__ == "__main__":
    main()
