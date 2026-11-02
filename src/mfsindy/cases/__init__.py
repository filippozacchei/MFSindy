"""Case-specific utilities and experiment helpers."""

from importlib import import_module

__all__ = [
    "burgers",
    "double_pendulum_forecasting",
    "hopf",
    "isothermal_flow",
    "lorenz",
    "pendulum",
    "vorticity_flow",
]

for _name in list(__all__):
    globals()[_name] = import_module(f"{__name__}.{_name}")
