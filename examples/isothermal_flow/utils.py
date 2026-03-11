"""Data generation helpers for the compressible isothermal flow case."""

from mfsindy.cases import isothermal_flow as _iso

make_initial_condition = _iso.make_initial_condition
generate_isothermal_ns_dataset = _iso.generate_isothermal_ns_dataset

__all__ = [
    "make_initial_condition",
    "generate_isothermal_ns_dataset",
]
