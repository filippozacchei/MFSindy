"""The multi-fidelity data of Part I: separate trajectories for an ODE, separate
windows of one flow for a PDE.

Two figures, for subfigures in the paper, with no text beyond axis labels.
``mf_data_ode``: the Lorenz attractor in grey with the noisy samples of one HF and
two LF trajectories of a run, and their x, y, z time series. ``mf_data_pde``: one
Burgers solution as a grey surface with one HF and two LF window strips coloured,
each connected to the noisy spatial profiles it observes. Everything is drawn
through the generators the Monte Carlo uses, so the figures show the data the
identification sees.

Lorenz shows run 0. For Burgers the windows are placed at random and about half of
them miss the flow, so the first run is shown whose HF window covers the peak and
whose two LF windows see the flow with little overlap; the caption should say so.
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
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import proj3d

from mfsindy.cases import burgers as B
from mfsindy.cases import lorenz as L
from mfsindy.plots import _RC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

HF, LF, GREY = "#2f6fae", "#c0392b", "0.55"

#: A window "sees" the flow when its largest |u| is at least this share of the
#: flow's; the HF window must reach PEAK of it, so that it covers the bump.
SEEN, PEAK = 0.3, 0.95
#: Largest share of a window the shown LF windows may have in common with the HF
#: window or with each other.
OVERLAP = 0.25


def make_config(case: str):
    """The Part I config of ``examples/<case>/main1.py``, as the Monte Carlo builds it."""

    path = ROOT / "examples" / case / "main1.py"
    spec = importlib.util.spec_from_file_location(f"{case}_main1", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module.make_config()


def lorenz_data():
    cfg = make_config("lorenz")
    std = L._lorenz_reference_state_std(cfg)
    noisy = L._lorenz_batch(0, cfg, cfg.noise_hf_rel * std, cfg.noise_lf_rel * std)
    clean = L._lorenz_batch(0, cfg, 0.0, 0.0)
    (attractor,), _, _ = L.generate_lorenz_dataset(n_traj=1, T=50.0, dt=cfg.dt, noise_level=0.0, seed=7)
    trajectories = [(np.asarray(clean.hf[0]), np.asarray(noisy.hf[0]), HF, "HF")]
    trajectories += [(np.asarray(clean.lf[k]), np.asarray(noisy.lf[k]), LF, "LF") for k in range(2)]
    t = np.arange(trajectories[0][0].shape[0]) * cfg.dt
    return attractor, trajectories, t


def burgers_data():
    cfg = make_config("burgers")
    std = B._burgers_reference_state_std(cfg)
    for run in range(cfg.n_runs):
        batch = B._burgers_batch(run, cfg, cfg.noise_hf_rel * std, cfg.noise_lf_rel * std)
        U = batch.metadata["flow"]
        idx = [set(np.asarray(i).tolist()) for i in batch.metadata["windows"]]
        share = [np.abs(U[sorted(i)]).max() / np.abs(U).max() for i in idx]
        if share[0] < PEAK:
            continue
        chosen = [0]
        for k in range(1, len(idx)):
            if share[k] >= SEEN and all(len(idx[k] & idx[j]) <= OVERLAP * len(idx[k]) for j in chosen):
                chosen.append(k)
        if len(chosen) >= 3:
            break
    else:
        raise RuntimeError("no run has an HF window on the peak and two distinct LF windows on the flow")
    windows = [(0, HF)] + [(k, LF) for k in chosen[1:3]]
    return cfg, run, batch, windows


def strip_3d(ax, *labels: str) -> None:
    """Keep the three axes and their labels; drop grid, panes and ticks."""

    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.set_ticks([])
    for set_label, text in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), labels):
        set_label(text, labelpad=-12)


def zoom_column(fig, spec, n: int):
    sub = spec.subgridspec(n, 1, hspace=0.3)
    axes = [fig.add_subplot(sub[k]) for k in range(n)]
    for a in axes:
        # Ticks and labels on the right, so the arrows arrive on a clear left edge.
        a.yaxis.tick_right()
        # tick_right draws the labels with the right-hand tick style, which the
        # rc sizes do not reach, so set them here.
        a.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
        a.yaxis.set_label_position("right")
        a.spines[["top", "left"]].set_visible(False)
    for a in axes[:-1]:
        a.set_xticklabels([])
    return axes


def to_figure(fig, transform, point):
    return fig.transFigure.inverted().transform(transform.transform(point))


def legend(ax, *extra) -> None:
    """HF and LF markers above a zoom column, plus any extra handles."""

    handles = [Line2D([], [], ls="none", marker=".", color=HF, label="HF"),
               Line2D([], [], ls="none", marker=".", color=LF, label="LF"), *extra]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=len(handles),
              frameon=False, handletextpad=0.1, columnspacing=1.0, borderaxespad=0.0)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def ode_figure() -> None:
    attractor, trajectories, t = lorenz_data()
    fig = plt.figure(figsize=(7.2, 3.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.45)

    ax = fig.add_subplot(grid[0], projection="3d")
    ax.plot(*attractor.T, color=GREY, lw=0.3, alpha=0.25)
    for _, noisy, color, _ in trajectories:
        ax.plot(*noisy[::5].T, ".", ms=1.0, color=color, alpha=0.7, rasterized=True)
    strip_3d(ax, "$x$", "$y$", "$z$")

    # One panel per state component, each with all three trajectories, the LF
    # ones under the HF one.
    zooms = zoom_column(fig, grid[1], 3)
    for d, (a, name) in enumerate(zip(zooms, "xyz")):
        for _, noisy, color, _ in trajectories[::-1]:
            a.plot(t, noisy[:, d], ".", ms=0.6, color=color, alpha=0.6, rasterized=True)
        a.set_xlim(t[0], t[-1])
        a.set_ylabel(f"${name}$")
    zooms[-1].set_xlabel("$t$")
    legend(zooms[0])

    fig.canvas.draw()
    end = to_figure(fig, zooms[1].transAxes, (-0.04, 0.5))
    begin = (to_figure(fig, ax.transAxes, (1.0, 0.0))[0], end[1])
    fig.add_artist(ConnectionPatch(xyA=begin, coordsA=fig.transFigure, xyB=end, coordsB=fig.transFigure,
                                   arrowstyle="-|>,head_width=0.25", color="0.4", lw=1.2))
    save(fig, "mf_data_ode")


def pde_figure() -> None:
    cfg, run, batch, windows = burgers_data()
    print(f"Burgers: run {run}")
    U = batch.metadata["flow"]
    x = batch.metadata["x_flow"]
    t = np.arange(U.shape[1]) * cfg.dt
    vmax = np.abs(U).max()
    data = batch.hf + batch.lf

    fig = plt.figure(figsize=(7.2, 3.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.45)

    # The clean solution as a grey reference surface, every 10th time step, with
    # each window's strip lightly tinted and its noisy samples on it, as the
    # attractor and the trajectories for the ODE. HF last, so that where windows
    # overlap its tint stays visible.
    step = 10
    T_, X_ = np.meshgrid(t[::step], x, indexing="xy")
    faces = np.full(X_.shape + (4,), matplotlib.colors.to_rgba(GREY, 0.25))
    for w, color in windows[::-1]:
        faces[np.asarray(batch.metadata["windows"][w])] = matplotlib.colors.to_rgba(color, 0.25)
    ax = fig.add_subplot(grid[0], projection="3d")
    ax.plot_surface(X_, T_, U[:, ::step], facecolors=faces, linewidth=0, shade=False,
                    rstride=1, cstride=1, rasterized=True)
    for w, color in windows[::-1]:
        idx = np.asarray(batch.metadata["windows"][w])
        Tw, Xw = np.meshgrid(t[::50], x[idx][::2], indexing="xy")
        ax.plot(Xw.ravel(), Tw.ravel(), np.asarray(data[w])[::2, ::50, 0].ravel(), ".", ms=1.0,
                color=color, alpha=0.8, rasterized=True)
    strip_3d(ax, "$x$", "$t$", "$u$")
    ax.view_init(elev=28, azim=-50)

    # Each window's data as spatial profiles at t = 0, T/2, T, dark to light:
    # the noisy samples over the clean profile. Panels follow the order of the
    # strips on the surface, top to bottom, so the arrows do not cross.
    fig.canvas.draw()

    def start(w):
        idx = np.asarray(batch.metadata["windows"][w])
        i, n = idx[len(idx) // 2], len(t) - 1
        return proj3d.proj_transform(x[i], t[n], U[i, n], ax.get_proj())[:2]

    windows = sorted(windows, key=lambda win: -start(win[0])[1])
    zooms = zoom_column(fig, grid[1], len(windows))
    dx = x[1] - x[0]
    x_window = np.arange(len(batch.metadata["windows"][0])) * dx
    for a, (w, color) in zip(zooms, windows):
        idx = np.asarray(batch.metadata["windows"][w])
        observed = np.asarray(data[w])[..., 0]
        for n, alpha in zip((0, len(t) // 2, len(t) - 1), (1.0, 0.6, 0.3)):
            a.plot(x_window, U[idx, n], color="0.3", lw=0.6, alpha=alpha)
            a.plot(x_window, observed[:, n], ".", ms=1.5, color=color, alpha=alpha)
        a.set_xlim(x_window[0], x_window[-1])
        a.set_ylim(-0.15 * vmax, 1.15 * vmax)
    zooms[1].set_ylabel("$u$")
    zooms[-1].set_xlabel("$x-x_0$")
    legend(zooms[0], Line2D([], [], color="0.3", lw=0.6, label="clean"))

    for a, (w, color) in zip(zooms, windows):
        fig.add_artist(ConnectionPatch(xyA=start(w), coordsA=ax.transData, xyB=(-0.02, 0.5),
                                       coordsB=a.transAxes, arrowstyle="-|>,head_width=0.2",
                                       connectionstyle="arc3,rad=-0.15", color=color, lw=0.8))
    save(fig, "mf_data_pde")


def main() -> None:
    # The package's figure style, so these match the other figures.
    with plt.rc_context({**_RC, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7}):
        ode_figure()
        pde_figure()


if __name__ == "__main__":
    main()
