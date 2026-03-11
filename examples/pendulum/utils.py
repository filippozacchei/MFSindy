"""Pendulum trajectory/data generators for the notebooks."""

from mfsindy.cases import pendulum as _pendulum

simulate_pendulum_trajectory = _pendulum.simulate_pendulum_trajectory
generate_pendulum_dataset = _pendulum.generate_pendulum_dataset

__all__ = [
    "simulate_pendulum_trajectory",
    "generate_pendulum_dataset",
]
