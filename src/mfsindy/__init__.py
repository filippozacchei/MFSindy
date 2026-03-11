"""Core Python package for the weighted weak SINDy toolkit."""

from . import cases
from .weighted_weak_pde_library import WeightedWeakPDELibrary

__all__ = [
    "cases",
    "WeightedWeakPDELibrary",
]
