"""Core Python package for the weighted weak SINDy toolkit."""

from . import cases
from .metrics import absolute_deviation, ensemble_disagreement
from .plotting import plot_heatmap, set_paper_style
from .training import copy_sindy, run_ensemble_sindy
from .weighted_weak_pde_library import WeightedWeakPDELibrary

__all__ = [
    "cases",
    "absolute_deviation",
    "ensemble_disagreement",
    "plot_heatmap",
    "set_paper_style",
    "copy_sindy",
    "run_ensemble_sindy",
    "WeightedWeakPDELibrary",
]
