"""Data-generator shortcuts for the Burgers case notebooks."""

from mfsindy.cases import burgers as _burgers

make_space_time_grid = _burgers.make_space_time_grid
burgers_solver = _burgers.burgers_solver
random_initial_condition = _burgers.random_initial_condition
generate_burgers_dataset = _burgers.generate_burgers_dataset

__all__ = [
    "make_space_time_grid",
    "burgers_solver",
    "random_initial_condition",
    "generate_burgers_dataset",
]
