"""Plotting helpers shared across examples."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

__all__ = ["bubble_hist", "tuning_heatmap"]

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
    height = float(figsize[1])
    fig.subplots_adjust(
        left=0.25,
        right=0.98,
        bottom=0.60 / height,
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


def tuning_heatmap(
    score_table,
    *,
    selections: Mapping[str, object] | None = None,
    rungs: Iterable[str] | None = None,
    x: str = "stlsq_threshold",
    y: str = "H_xt",
    score_col: str = "score",
    rung_col: str = "rung",
    admissible_col: str = "admissible",
    ncols: int = 4,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = True,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """Show the whole tuning grid per rung, with the selected cell outlined.

    The selection is one number per rung; the surface behind it is what says
    whether that number means anything. A rung whose grid is flat was not really
    tuned, one whose best sits in a spike will not survive a reseed, and one
    whose best lies on an edge is asking for a wider grid -- none of which is
    visible from the chosen value alone. Cells that failed to fit are left
    blank rather than plotted as a low score, so a hole reads as a hole.

    Where a rung was restricted -- the covariance-weighted rungs are confined to
    supports on which their covariance model holds -- the cells it could not be
    selected from are still scored and still drawn, crosshatched. Showing the
    surface and then ruling part of it out is the point: a reader can see what
    the restriction cost, which a quietly shrunken grid would hide.

    ``score_table`` is the frame :func:`mfsindy.experiments.tune_rungs` returns;
    ``selections`` its companion mapping, used only to mark the chosen cell.
    """

    import matplotlib.patches as mpatches

    table = score_table.copy()
    for column in (x, y):
        table[column] = table[column].map(_grid_label)

    model_names = list(rungs) if rungs is not None else list(
        dict.fromkeys(table[rung_col])
    )
    x_levels = list(dict.fromkeys(table[x]))
    y_levels = list(dict.fromkeys(table[y]))

    finite = table[score_col].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if vmin is None:
        vmin = float(np.min(finite)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.max(finite)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9

    ncols = max(1, min(int(ncols), len(model_names)))
    nrows = int(np.ceil(len(model_names) / ncols))
    if figsize is None:
        figsize = (3.1 * ncols + 1.2, 2.5 * nrows + 0.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    mesh = None
    for idx, name in enumerate(model_names):
        ax = axes[idx // ncols][idx % ncols]
        sub = table[table[rung_col] == name]
        grid = (
            sub.pivot_table(index=y, columns=x, values=score_col, aggfunc="mean")
            .reindex(index=y_levels, columns=x_levels)
        )
        values = np.ma.masked_invalid(grid.to_numpy(dtype=float))
        mesh = ax.pcolormesh(
            np.arange(len(x_levels) + 1),
            np.arange(len(y_levels) + 1),
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if annot:
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    if values.mask[row, col] if np.ma.is_masked(values) else False:
                        continue
                    value = float(values[row, col])
                    shade = (value - vmin) / (vmax - vmin)
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if shade < 0.55 else "black",
                    )

        if admissible_col in sub.columns:
            blocked = (
                sub.pivot_table(
                    index=y, columns=x, values=admissible_col, aggfunc="min"
                )
                .reindex(index=y_levels, columns=x_levels)
                .to_numpy(dtype=float)
            )
            for row in range(blocked.shape[0]):
                for col in range(blocked.shape[1]):
                    if blocked[row, col] == 0.0:
                        ax.add_patch(
                            mpatches.Rectangle(
                                (col, row), 1, 1,
                                fill=False, hatch="xxx",
                                edgecolor="0.25", linewidth=0.0,
                            )
                        )

        chosen = None if selections is None else selections.get(name)
        best = getattr(chosen, "best_params", None)
        if best is not None and x in best and y in best:
            try:
                col = x_levels.index(_grid_label(best[x]))
                row = y_levels.index(_grid_label(best[y]))
            except ValueError:
                col = row = None
            if col is not None:
                ax.add_patch(
                    mpatches.Rectangle(
                        (col, row), 1, 1, fill=False, edgecolor="red", linewidth=2.0
                    )
                )

        ax.set_xticks(np.arange(len(x_levels)) + 0.5)
        ax.set_yticks(np.arange(len(y_levels)) + 0.5)
        ax.set_xticklabels(x_levels, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(y_levels, fontsize=7)
        ax.set_title(name, fontsize=10)
        if idx % ncols == 0:
            ax.set_ylabel(y, fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(x, fontsize=8)

    for idx in range(len(model_names), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label=score_col, shrink=0.85)
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(f"{save_path}.{extension}", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
