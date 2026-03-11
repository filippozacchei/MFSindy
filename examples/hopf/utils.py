"""Data-generator shortcuts for the Hopf bifurcation case."""

from mfsindy.cases import hopf as _hopf

generate_hopf_trajectory = _hopf.generate_hopf_trajectory
generate_hopf_dataset = _hopf.generate_hopf_dataset

__all__ = [
    "generate_hopf_trajectory",
    "generate_hopf_dataset",
]
