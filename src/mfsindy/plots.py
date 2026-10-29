"""Plotting helpers shared across examples."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

__all__ = [
    "DISPLAY",
    "PART1_COLORS",
    "PART2_COLORS",
    "bubble_hist",
    "plot_coefficient_ensembles",
    "plot_coefficient_runs",
    "plot_run_trajectories",
    "plot_selection",
    "plot_state_noise",
    "plot_threshold_path",
]

def bubble_hist(
    errors_dict: Mapping[str, Iterable[float]],
    *,
    n_bins: int = 8,
    shared_bins: bool = False,
    log_x: bool = False,
    models_order: list[str] | tuple[str, ...] | None = None,
    colors: Dict[str, str] | None = None,
    labels: Iterable[str] | None = None,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    max_size: float = 520.0,
    alpha: float = 0.75,
    xlabel: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot one compact 1D bubble histogram with fixed dimensions.

    Each row is binned over its own range into ``n_bins`` bins, and scaled by its
    own peak count, so rows whose errors differ by orders of magnitude are each
    resolved. Pass ``shared_bins=True`` for one common set of bins across rows,
    where bubble areas compare directly between rows but a row much tighter than
    the widest one collapses into a couple of bins.

    ``log_x`` bins geometrically and sets a log axis. Per-row bins fix the
    resolution within a row, but the axis is still shared, so on a linear scale a
    single outlying row stretches it and leaves the accurate rows in a sliver at
    the left. Errors spanning orders of magnitude read better this way. It
    requires strictly positive values, so it does not suit the support metric,
    which is legitimately zero when a model recovers the support exactly.
    """

    sns.set_theme(style="white", context="paper")

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    if models_order is None:
        models = list(errors_dict.keys())
    else:
        models = list(models_order)

    if not models:
        raise ValueError("bubble_hist requires at least one model.")

    if colors is None:
        palette = sns.color_palette("tab10", n_colors=len(models))
        color_map = {m: palette[i] for i, m in enumerate(models)}
    else:
        color_map = colors

    arrays = [np.asarray(errors_dict[m], dtype=float) for m in models]

    if any(arr.size == 0 for arr in arrays):
        raise ValueError("Each model must contain at least one error value.")

    all_vals = np.concatenate(arrays)

    if xlim is None:
        vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
        if log_x:
            # Pad multiplicatively; an additive pad can cross zero on a log axis.
            xlim = (vmin / 1.6, vmax * 1.6)
        else:
            pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
            xlim = (vmin - pad, vmax + pad)

    # Bin each row over its own values rather than over one range shared by every
    # row. The rows span orders of magnitude -- a rung with errors near 0.02 sits
    # beside one near 1.5 -- so shared bins collapse the accurate rows into the
    # first bin or two and hide the spread that distinguishes them.
    if log_x and float(np.min(all_vals)) <= 0.0:
        raise ValueError(
            "log_x needs strictly positive values; this metric reaches "
            f"{float(np.min(all_vals))}."
        )

    def spaced(low: float, high: float) -> np.ndarray:
        if log_x:
            return np.geomspace(low, high, n_bins + 1)
        return np.linspace(low, high, n_bins + 1)

    def row_bins(values: np.ndarray) -> np.ndarray:
        low, high = float(np.min(values)), float(np.max(values))
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError("Error values must be finite to be binned.")
        if high <= low:
            # A row of one repeated value (exact support recovery, say) gets a
            # narrow bin centred on it rather than an empty range.
            pad = 0.005 * (abs(low) if low else 1.0)
            low, high = low - pad, high + pad
        return spaced(low, high)

    if shared_bins:
        shared = spaced(*xlim)
        bins_per_model = {m: shared for m in models}
    else:
        bins_per_model = {m: row_bins(arr) for m, arr in zip(models, arrays)}

    if figsize is None:
        # Grow the panel with the number of rows so bubbles do not collide.
        # Four rows or fewer keep the original height.
        figsize = (3.2, 2.0 * max(1.0, len(models) / 4.0))
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    counts_per_model = {}
    centers_per_model = {}
    max_count = 0

    for m, arr in zip(models, arrays):
        edges = bins_per_model[m]
        counts, _ = np.histogram(arr, bins=edges)
        counts_per_model[m] = counts
        centers_per_model[m] = (
            np.sqrt(edges[:-1] * edges[1:]) if log_x else 0.5 * (edges[:-1] + edges[1:])
        )
        max_count = max(max_count, int(counts.max(initial=0)))

    for idx, m in enumerate(models):
        counts = counts_per_model[m]
        centers = centers_per_model[m]
        # With per-row bins the bin widths differ between rows, so a count in one
        # row is not comparable with a count in another; scale each row by its own
        # peak so every row shows its own shape. Shared bins keep the common
        # scaling, where counts do compare.
        reference = max_count if shared_bins else int(counts.max(initial=0))
        sizes = max_size * counts / max(1, reference)

        ax.scatter(
            centers,
            np.full_like(centers, idx),
            s=sizes,
            color=color_map.get(m, "gray"),
            alpha=alpha,
            edgecolors="black",
            linewidths=0.4,
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(-0.6, len(models) - 0.4)

    ax.set_yticks(range(len(models)))

    if labels is None:
        ax.set_yticklabels(models)
    else:
        ax.set_yticklabels(list(labels))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=4)

    # Stable margins: useful when arranging manually in Keynote. The vertical
    # margins are fixed in inches rather than as a fraction of the figure, so a
    # taller panel adds plotting area instead of white space.
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    height = float(figsize[1])
    fig.subplots_adjust(
        left=0.25,
        right=0.93,                       # room for a log tick at the right edge
        bottom=(0.95 if xlabel is not None else 0.60) / height,
        top=1.0 - 0.10 / height,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches=None, transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _grid_label(value) -> str:
    """Short axis label for one grid value, including the PDE list-valued ones."""

    if isinstance(value, (list, tuple, np.ndarray)):
        return "(" + ", ".join(f"{float(v):g}" for v in np.ravel(value)) + ")"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


#: Figure style, fixed inside each figure so a notebook's seaborn context or an
#: earlier bubble_hist call does not leak in: 7 pt sans-serif, thin lines.
_RC = {"font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7,
       "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 7,
       "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
       "xtick.major.size": 2, "ytick.major.size": 0, "axes.spines.top": False,
       "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}

#: Part I models coloured by family: single fidelity unweighted (greys) and
#: weighted (blues), fusion without fidelity weights (oranges), and fusion with
#: them (teal for the per-group variances, red for the full weak covariance).
PART1_COLORS = {"HF": "#5f5f5f", "LF": "#a9a9a9", "VHF": "#2f6fae", "VLF": "#8db5dc",
                "MF": "#d98c00", "PMF": "#f2c46b", "VMF": "#2a9d8f", "MF_w": "#c0392b"}
PART2_COLORS = {"No weighting": "#8c8c8c", "Ones GLS": "#d98c00",
                "Variance diag": "#2a9d8f", "Variance GLS": "#c0392b"}
#: Display names, as the paper writes them.
DISPLAY = {"MF_w": "WMF"}

_GREEK = {"theta": r"\theta", "omega": r"\omega", "rho": r"\rho", "phi": r"\phi"}


def _tex(name: str) -> str:
    """A library or state name as mathtext: Greek words become symbols."""

    out = str(name)
    for word, symbol in _GREEK.items():
        out = out.replace(word, symbol)
    return f"${out}$"


def _dot(name: str) -> str:
    return rf"$\dot{{{_tex(name)[1:-1]}}}$"


def _label(name: str) -> str:
    return DISPLAY.get(name, name)


def _finish(fig, save_path: str | None, show: bool) -> None:
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=True)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_selection(
    record: Mapping,
    *,
    rungs: list[str] | tuple[str, ...],
    colors: Mapping[str, str] | None = None,
    highlight: Iterable[str] = ("MF_w", "VMF", "Variance GLS", "Variance diag"),
    equation_names: Iterable[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """How each rung chose its support and its threshold, for one run.

    Left: the spread of the bagged coefficients against the candidate support,
    the chosen support filled. Then one panel per equation: the rung's
    fit-complexity Pareto front at its chosen support -- the best RSS reachable
    with at most that many terms, relative to the rung's best fit, since whitened
    and plain systems carry different units -- with the BIC choice filled. BIC picks a point on this
    front, with the trade-off fixed by log N. Rungs in ``highlight`` are drawn
    full strength, the others thin and light.
    """

    colors = dict(colors or {})
    highlight = set(highlight)
    by_rung = record["rungs"]
    supports = list(by_rung[rungs[0]]["by_support"])
    numeric = all(np.ndim(H) == 0 for H in supports)
    xs = np.asarray(supports, dtype=float) if numeric else np.arange(len(supports))
    n_eq = np.asarray(by_rung[rungs[0]]["by_support"][supports[0]]["rss"]).shape[1]
    names = [_dot(n) for n in equation_names] if equation_names is not None else [f"eq. {j}" for j in range(n_eq)]

    def style(rung):
        strong = rung in highlight
        return dict(color=colors.get(rung, "k"), lw=1.2 if strong else 0.7,
                    alpha=1.0 if strong else 0.55, zorder=3 if strong else 2)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 1 + n_eq, figsize=(1.75 * (1 + n_eq) + 0.9, 1.7))
        ax = axes[0]
        for rung in rungs:
            rec = by_rung[rung]
            spread = np.array([rec["by_support"][H]["spread"] for H in supports], dtype=float)
            st = style(rung)
            ax.plot(xs, spread, "-", label=_label(rung), **st)
            k = supports.index(rec["support"])
            ax.plot(xs[k], spread[k], "o", ms=3.2, color=st["color"], alpha=st["alpha"], zorder=4)
        if numeric:
            ax.set_xscale("log")
        else:
            ax.set_xticks(xs, [_grid_label(H) for H in supports], rotation=90)
        ax.set_yscale("log")
        ax.set_xlabel("support $H$")
        ax.set_ylabel("coefficient spread")
        ax.set_title("support")

        for j, ax in enumerate(axes[1:]):
            for rung in rungs:
                rec = by_rung[rung]["by_support"][by_rung[rung]["support"]]
                nnz, rss = np.asarray(rec["nnz"])[:, j], np.asarray(rec["rss"])[:, j]
                rel = rss / max(rss.min(), 1e-300)
                # The front: the best fit reachable with at most that many terms
                # (the non-dominated points of the threshold path).
                sizes = np.unique(nnz)
                front = np.minimum.accumulate([rel[nnz == n].min() for n in sizes])
                st = style(rung)
                ax.plot(sizes, front, "-", **st)
                i = int(np.asarray(rec["choice"])[j])
                ax.plot(nnz[i], rel[i], "o", ms=3.2, color=st["color"], alpha=st["alpha"], zorder=4)
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlabel("terms kept")
            ax.set_title(f"threshold (BIC): {names[j]}")
        axes[1].set_ylabel("RSS / min RSS")
        for ax in axes:
            ax.tick_params(axis="y", length=2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
                   handlelength=1.4)
        fig.tight_layout(w_pad=1.0)
        _finish(fig, save_path, show)


def plot_threshold_path(
    record: Mapping,
    *,
    rungs: list[str] | tuple[str, ...],
    colors: Mapping[str, str] | None = None,
    highlight: Iterable[str] = ("MF_w", "VMF", "Variance GLS", "Variance diag"),
    equation_names: Iterable[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """The whole threshold sweep behind the BIC choice, for one run.

    One column per equation; along the x-axis every threshold tried, relative to
    the equation's largest least-squares coefficient. Rows: the terms the median
    model keeps, its RSS relative to the rung's best fit, and its BIC relative to
    the minimum. Each rung is shown at the support it chose, and the threshold
    BIC picked is filled. :func:`plot_selection` shows the same sweep reduced to
    its Pareto front.
    """

    colors = dict(colors or {})
    highlight = set(highlight)
    by_rung = record["rungs"]
    first = by_rung[rungs[0]]["by_support"][by_rung[rungs[0]]["support"]]
    n_eq = np.asarray(first["rss"]).shape[1]
    names = [_dot(n) for n in equation_names] if equation_names is not None else [f"eq. {j}" for j in range(n_eq)]
    rows = (("nnz", "terms kept", False), ("rss", "RSS / min RSS", True), ("bic", "BIC $-$ min BIC", False))

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(3, n_eq, figsize=(1.9 * n_eq + 0.9, 4.2), sharex=True, squeeze=False)
        for j in range(n_eq):
            for rung in rungs:
                rec = by_rung[rung]["by_support"][by_rung[rung]["support"]]
                lam = np.asarray(rec["thresholds"])[:, j]
                lam = lam / lam.max()
                i = int(np.asarray(rec["choice"])[j])
                strong = rung in highlight
                st = dict(color=colors.get(rung, "k"), lw=1.2 if strong else 0.7,
                          alpha=1.0 if strong else 0.55, zorder=3 if strong else 2)
                for r, (key, _, _) in enumerate(rows):
                    y = np.asarray(rec[key], dtype=float)[:, j]
                    if key == "rss":
                        y = y / max(y.min(), 1e-300)
                    elif key == "bic":
                        y = y - y.min()
                    ax = axes[r, j]
                    ax.plot(lam, y, "-", label=_label(rung) if (r, j) == (0, 0) else None, **st)
                    ax.plot(lam[i], y[i], "o", ms=3.2, color=st["color"], alpha=st["alpha"], zorder=4)
            axes[0, j].set_title(names[j])
            axes[-1, j].set_xlabel("threshold / max coefficient")
            for r, (_, label, logy) in enumerate(rows):
                ax = axes[r, j]
                ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                elif r == 2:
                    ax.set_yscale("symlog", linthresh=10)
                    ax.set_ylim(bottom=-1)          # BIC - min BIC is never negative
                if r == 0:
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                if j == 0:
                    ax.set_ylabel(label)
                ax.tick_params(axis="y", length=2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
                   handlelength=1.4)
        fig.tight_layout(h_pad=0.6, w_pad=1.0)
        _finish(fig, save_path, show)


def plot_run_trajectories(
    clean: list,
    noisy: list,
    *,
    dims: tuple[int, int] = (0, 1),
    state_names: Iterable[str] | None = None,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
    reference_label: str = "",
    every: int = 5,
    color: str = "#2f6fae",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """The trajectories of one run in the plane of two state components.

    Each trajectory's noisy samples (every ``every``-th) and clean path, with its
    start marked. ``reference`` is an optional curve drawn dashed, such as a
    limit cycle.
    """

    i, j = dims
    d = np.asarray(clean[0]).shape[1]
    names = [_tex(n) for n in state_names] if state_names is not None else [f"$x_{k}$" for k in range(d)]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        for x, y in zip(clean, noisy):
            x, y = np.asarray(x), np.asarray(y)
            ax.plot(y[::every, i], y[::every, j], ".", ms=1.0, color=color, alpha=0.35, rasterized=True)
            ax.plot(x[:, i], x[:, j], "-", lw=0.8, color="k")
            ax.plot(x[0, i], x[0, j], "o", ms=2.5, color="k")
        if reference is not None:
            ax.plot(*reference, "--", lw=0.8, color="0.4", label=reference_label or None)
            if reference_label:
                ax.legend(loc="upper right", frameon=False, fontsize=6)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_aspect("equal", "datalim")
        fig.tight_layout()
        _finish(fig, save_path, show)


def plot_state_noise(
    t: np.ndarray,
    x_clean: np.ndarray,
    x_noisy: np.ndarray,
    sigma: np.ndarray,
    *,
    state_names: Iterable[str] | None = None,
    every: int = 10,
    color: str = "#2f6fae",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Each state component against time: clean, noisy, and its noise level.

    One panel per component, sharing the time axis: the noisy samples (every
    ``every``-th, so the trace stays readable), the clean signal, and a band of
    plus or minus two standard deviations of that component's noise, which shows
    where each component is quiet and where it is loud.
    """

    t = np.asarray(t, dtype=float)
    x_clean, x_noisy = np.asarray(x_clean, dtype=float), np.asarray(x_noisy, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.broadcast_to(sigma[:, None] if sigma.ndim == 1 else sigma, x_clean.shape)
    d = x_clean.shape[1]
    names = [_tex(n) for n in state_names] if state_names is not None else [f"$x_{i}$" for i in range(d)]

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(d, 1, figsize=(6.5, 0.95 * d + 0.4), sharex=True, squeeze=False)
        for i, ax in enumerate(axes[:, 0]):
            ax.fill_between(t, x_clean[:, i] - 2 * sigma[:, i], x_clean[:, i] + 2 * sigma[:, i],
                            color=color, alpha=0.18, lw=0, label=r"$\pm 2\sigma_i$")
            ax.plot(t[::every], x_noisy[::every, i], ".", ms=1.2, color=color, alpha=0.6,
                    label="noisy", rasterized=True)
            ax.plot(t, x_clean[:, i], "-", lw=0.8, color="k", label="clean")
            ax.set_ylabel(names[i])
            ax.tick_params(axis="y", length=2)
        axes[-1, 0].set_xlabel("$t$")
        axes[0, 0].legend(loc="upper right", ncol=3, frameon=False, fontsize=6, markerscale=4)
        fig.tight_layout(h_pad=0.3)
        _finish(fig, save_path, show)


def _coefficient_panels(
    samples: Mapping[str, np.ndarray],
    true: np.ndarray,
    *,
    rungs: list[str] | tuple[str, ...],
    colors: Mapping[str, str] | None,
    row_names: Iterable[str] | None,
    col_names: Iterable[str] | None,
    ncols: int,
    save_path: str | None,
    show: bool,
) -> None:
    """One small panel per true term, rungs as rows: every sample a dot, the
    median a tick, the true value a dashed line. ``samples[rung]`` is stacked
    over its first axis (bags of one run, or runs). Limits follow the
    interquartile ranges, so a rung that cannot identify the system runs off."""

    colors = dict(colors or {})
    true = np.asarray(true)
    cells = sorted(zip(*np.nonzero(true != 0)), key=lambda ij: (ij[1], ij[0]))
    rows = [_tex(n) for n in row_names] if row_names is not None else [f"term {i}" for i in range(true.shape[0])]
    cols = [_dot(n) for n in col_names] if col_names is not None else [f"eq. {j}" for j in range(true.shape[1])]
    order = list(rungs)[::-1]                        # first rung on top

    ncols = min(ncols, len(cells))
    nrows = int(np.ceil(len(cells) / ncols))
    rng = np.random.default_rng(0)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(1.45 * ncols + 0.4, (0.13 * len(rungs) + 0.45) * nrows),
                                 squeeze=False, sharey=True)
        for ax, (i, j) in zip(axes.ravel(), cells):
            quartiles = []
            for k, rung in enumerate(order):
                v = np.asarray(samples[rung])[:, i, j]
                col = colors.get(rung, "k")
                quartiles += list(np.percentile(v, [25, 75]))
                ax.scatter(v, k + 0.13 * rng.standard_normal(len(v)), s=1.5, color=col, alpha=0.45,
                           linewidths=0, rasterized=True)
                ax.plot([np.median(v)] * 2, [k - 0.36, k + 0.36], "-", lw=1.2, color=col)
            ax.axvline(true[i, j], ls=(0, (2, 2)), lw=0.6, color="k")
            lo, hi = min(quartiles + [true[i, j]]), max(quartiles + [true[i, j]])
            pad = 0.6 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1.0)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(-0.6, len(order) - 0.4)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.set_title(f"{cols[j]}: {rows[i]}")
            ax.spines["left"].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_yticks(range(len(order)), [_label(r) for r in order])
        for ax in axes.ravel()[len(cells):]:
            ax.axis("off")
        fig.tight_layout(w_pad=0.6, h_pad=0.8)
        _finish(fig, save_path, show)


def plot_coefficient_ensembles(
    record: Mapping,
    *,
    rungs: list[str] | tuple[str, ...],
    colors: Mapping[str, str] | None = None,
    row_names: Iterable[str] | None = None,
    col_names: Iterable[str] | None = None,
    ncols: int = 4,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """The bagged coefficients of every rung on one run (the last), per true term.

    Every ensemble member is a dot, the median -- the coefficient the rung
    reports -- a tick, the true value the dashed line. Spurious terms are left
    out; the front panels of :func:`plot_selection` account for them.
    """

    _coefficient_panels({r: record["rungs"][r]["bags"] for r in rungs}, record["true"], rungs=rungs,
                        colors=colors, row_names=row_names, col_names=col_names, ncols=ncols,
                        save_path=save_path, show=show)


def plot_coefficient_runs(
    record: Mapping,
    *,
    rungs: list[str] | tuple[str, ...],
    colors: Mapping[str, str] | None = None,
    row_names: Iterable[str] | None = None,
    col_names: Iterable[str] | None = None,
    ncols: int = 4,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """The coefficient every rung reported on every Monte Carlo run, per true term.

    The across-runs counterpart of :func:`plot_coefficient_ensembles`: each dot is
    one run's reported (median) coefficient, so the cloud is the estimator's
    sampling distribution -- its spread the variance, its offset from the dashed
    true value the bias.
    """

    if "coefficients" not in record:
        raise KeyError("This record has no per-run coefficients; rerun the Monte Carlo to store them.")
    _coefficient_panels(record["coefficients"], record["true"], rungs=rungs, colors=colors,
                        row_names=row_names, col_names=col_names, ncols=ncols,
                        save_path=save_path, show=show)
