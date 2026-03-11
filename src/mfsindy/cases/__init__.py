"""Case-specific utilities and experiment helpers."""

from importlib import import_module

__all__ = [
    "burgers",
    "hopf",
    "isothermal_flow",
    "lorenz",
    "pendulum",
]

for _name in list(__all__):
    globals()[_name] = import_module(f"{__name__}.{_name}")
