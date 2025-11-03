import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def set_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    })

def plot_heatmap(
    matrix,
    n_lf_vals,
    n_hf_vals,
    cmap="magma",
    fname=None,
    label=r"$R^2$",
):
    """
    Produce a publication-quality heatmap.
    """

    set_paper_style()

    fig, ax = plt.subplots(figsize=(3.5, 2.8))  # column width ≈ 90 mm

    # --- Plot heatmap ---
    im = ax.imshow(matrix, origin="lower", cmap=cmap, aspect="auto", interpolation="nearest")

    # --- Colorbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(label, rotation=270, labelpad=12)
    cbar.ax.tick_params(length=2, width=0.5)

    # --- Axes ticks and labels ---
    ax.set_xticks(np.arange(len(n_hf_vals)))
    ax.set_xticklabels(n_hf_vals)
    ax.set_yticks(np.arange(len(n_lf_vals)))
    ax.set_yticklabels(n_lf_vals)
    ax.set_xlabel(r"$n_{\mathrm{HF}}$")
    ax.set_ylabel(r"$n_{\mathrm{LF}}$")

    # --- Add gridlines for clarity ---
    ax.set_xticks(np.arange(-0.5, len(n_hf_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(n_lf_vals), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", length=0)

    # --- Layout and save ---
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
