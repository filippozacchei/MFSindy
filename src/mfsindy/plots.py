"""Plotting helpers shared across examples."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

__all__ = ["bubble_hist"]

def bubble_hist(
    errors_dict: Mapping[str, Iterable[float]],
    *,
    n_bins: int = 8,
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
    """Plot one compact 1D bubble histogram with fixed dimensions."""

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
        pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
        xlim = (vmin - pad, vmax + pad)

    bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    if figsize is None:
        # Grow the panel with the number of rows so bubbles do not collide.
        # Four rows or fewer keep the original height.
        figsize = (3.2, 2.0 * max(1.0, len(models) / 4.0))
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    counts_per_model = {}
    max_count = 0

    for m, arr in zip(models, arrays):
        counts, _ = np.histogram(arr, bins=bins)
        counts_per_model[m] = counts
        max_count = max(max_count, int(counts.max(initial=0)))

    for idx, m in enumerate(models):
        counts = counts_per_model[m]
        sizes = max_size * counts / max(1, max_count)

        ax.scatter(
            centers,
            np.full_like(centers, idx),
            s=sizes,
            color=color_map.get(m, "gray"),
            alpha=alpha,
            edgecolors="black",
            linewidths=0.4,
        )

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
