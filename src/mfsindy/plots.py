"""Plotting helpers shared across examples."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

__all__ = ["bubble_hist"]


def bubble_hist(
    errors_dict: Mapping[str, Iterable[float]],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    n_bins: int = 8,
    models_order: list[str] | tuple[str, ...] | None = None,
    colors: Dict[str, str] | None = None,
    labels: Iterable[str] | None = None,
) -> None:
    """Plot compact 1D bubble histograms for a collection of methods."""

    sns.set(style="white", context="paper")

    if models_order is None:
        models = list(errors_dict.keys())
    else:
        models = list(models_order)

    if not models:
        raise ValueError("bubble_hist requires at least one model.")

    color_map: Dict[str, str]
    if colors is None:
        palette = sns.color_palette("tab10", n_colors=len(models))
        color_map = {m: palette[i] for i, m in enumerate(models)}
    else:
        color_map = colors

    arrays = [np.asarray(errors_dict[m], dtype=float) for m in models]
    all_vals = np.concatenate(arrays)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    bins = np.linspace(vmin - pad, vmax + pad, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(3, 1.5), dpi=300)

    max_count = 0
    counts_per_model = {}
    for m, arr in zip(models, arrays):
        counts, _ = np.histogram(arr, bins=bins)
        counts_per_model[m] = counts
        max_count = max(max_count, int(counts.max(initial=0)))

    for idx, m in enumerate(models):
        counts = counts_per_model[m]
        sizes = 150.0 * counts / max(1, max_count)
        ax.scatter(
            centers,
            np.full_like(centers, idx),
            s=sizes,
            color=color_map.get(m, "gray"),
            alpha=0.7,
        )

    ax.set_yticks(range(len(models)))
    if labels is None:
        ax.set_yticklabels(models)
    else:
        ax.set_yticklabels(list(labels))
    ax.set_xlabel(xlabel or "")
    ax.set_title(title or "", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()

