"""Base diffusion demo reuses the Burgers data generators."""

from mfsindy.cases import burgers as _burgers

make_space_time_grid = _burgers.make_space_time_grid
burgers_solver = _burgers.burgers_solver
random_initial_condition = _burgers.random_initial_condition

__all__ = [
    "make_space_time_grid",
    "burgers_solver",
    "random_initial_condition",
]
