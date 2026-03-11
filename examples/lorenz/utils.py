"""Convenience exports for Lorenz trajectory generation."""

from mfsindy.cases import lorenz as _lorenz

lorenz = _lorenz.lorenz
generate_lorenz_trajectory = _lorenz.generate_lorenz_trajectory
generate_lorenz_dataset = _lorenz.generate_lorenz_dataset

__all__ = [
    "lorenz",
    "generate_lorenz_trajectory",
    "generate_lorenz_dataset",
]
