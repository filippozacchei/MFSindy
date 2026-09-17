"""Rollout-based validation helpers for ODE-style SINDy examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.feature_library import WeakPDELibrary

from mfsindy.weighted_weak_pde_library import (
    DedupedWeakPDELibrary,
    WeightedWeakPDELibrary,
)

from .intra_trajectory import IntraTrajectoryGLSData


def split_trajectory_list(
    trajectories: Sequence[np.ndarray],
    *,
    validation_fraction: float = 0.2,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split a list of trajectories into train/validation subsets."""

    trajectories = [np.asarray(traj, dtype=float) for traj in trajectories]
    n_total = len(trajectories)
    if n_total < 2:
        raise ValueError("At least two trajectories are required for a train/validation split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1).")

    n_val = max(1, int(np.ceil(n_total * validation_fraction)))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val
    return trajectories[:n_train], trajectories[n_train:]


def split_single_trajectory(
    trajectory: np.ndarray,
    t_grid: np.ndarray,
    *,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split one trajectory into contiguous train and validation windows."""

    trajectory = np.asarray(trajectory, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape (n_steps, n_states).")
    if trajectory.shape[0] != t_grid.shape[0]:
        raise ValueError("trajectory and t_grid must have the same number of time samples.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1).")

    n_total = trajectory.shape[0]
    n_val = max(2, int(np.ceil(n_total * validation_fraction)))
    n_val = min(n_val, n_total - 2)
    n_train = n_total - n_val

    train_traj = trajectory[:n_train].copy()
    val_traj = trajectory[n_train - 1 :].copy()

    train_t = t_grid[:n_train] - t_grid[0]
    val_t = t_grid[n_train - 1 :] - t_grid[n_train - 1]
    return train_traj, val_traj, train_t, val_t


def rollout_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute a multivariate rollout R^2 score over time and state dimensions."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}.")
    if not np.all(np.isfinite(y_pred)):
        return -1e12

    residual = y_true - y_pred
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    denom = float(np.sum(centered**2))
    if denom <= 0.0:
        return -1e12
    return float(1.0 - np.sum(residual**2) / denom)


@dataclass
class PolynomialRolloutModel:
    """Lightweight polynomial ODE model used for rollout scoring and equation display."""

    coefficients: np.ndarray
    library: Any
    state_names: tuple[str, ...]
    feature_names: tuple[str, ...]

    def _theta(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        theta = self.library.transform([state[None, :]])[0]
        return np.asarray(theta, dtype=float)[0]

    def rhs(self, state: np.ndarray) -> np.ndarray:
        theta = self._theta(state)
        return theta @ self.coefficients.T

    def simulate(self, x0: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float)
        t_grid = np.asarray(t_grid, dtype=float)
        if t_grid.ndim != 1 or t_grid.size < 2:
            raise ValueError("t_grid must be a 1D array with at least two time samples.")

        out = np.zeros((t_grid.size, x0.size), dtype=float)
        out[0] = x0
        for idx in range(1, t_grid.size):
            dt = float(t_grid[idx] - t_grid[idx - 1])
            state = out[idx - 1]
            k1 = self.rhs(state)
            k2 = self.rhs(state + 0.5 * dt * k1)
            k3 = self.rhs(state + 0.5 * dt * k2)
            k4 = self.rhs(state + dt * k3)
            out[idx] = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.all(np.isfinite(out[idx])):
                out[idx:] = np.nan
                break
        return out

    def equations(self, *, precision: int = 3, tol: float = 1e-10) -> list[str]:
        lines: list[str] = []
        for state_name, row in zip(self.state_names, self.coefficients, strict=True):
            terms: list[str] = []
            for coef, feature_name in zip(row, self.feature_names, strict=True):
                coef = float(coef)
                if abs(coef) <= tol:
                    continue
                sign = "-" if coef < 0 else "+"
                magnitude = abs(coef)
                if feature_name == "1":
                    term = f"{magnitude:.{precision}g}"
                else:
                    term = f"{magnitude:.{precision}g} {feature_name}"
                terms.append((sign, term))

            if not terms:
                rhs = "0"
            else:
                first_sign, first_term = terms[0]
                rhs_parts = [first_term if first_sign == "+" else f"-{first_term}"]
                rhs_parts.extend(f" {sign} {term}" for sign, term in terms[1:])
                rhs = "".join(rhs_parts)

            lines.append(f"{state_name}_dot = {rhs}")
        return lines


def build_polynomial_rollout_models(
    coefficient_map: Mapping[str, np.ndarray],
    *,
    poly_degree: int,
    reference_trajectory: np.ndarray,
    state_names: Sequence[str],
    include_bias: bool = False,
) -> dict[str, PolynomialRolloutModel]:
    """Create rollout-capable polynomial models from a map of coefficient matrices."""

    state_names = tuple(state_names)
    reference_trajectory = np.asarray(reference_trajectory, dtype=float)
    if reference_trajectory.ndim != 2:
        raise ValueError("reference_trajectory must have shape (n_steps, n_states).")

    library = ps.PolynomialLibrary(degree=poly_degree, include_bias=include_bias)
    library.fit([reference_trajectory])
    feature_names = tuple(library.get_feature_names(input_features=list(state_names)))

    return {
        model_name: PolynomialRolloutModel(
            coefficients=np.asarray(coefficients, dtype=float),
            library=library,
            state_names=state_names,
            feature_names=feature_names,
        )
        for model_name, coefficients in coefficient_map.items()
    }


def build_polynomial_intra_trajectory_artifacts(
    data: np.ndarray,
    t_grid: np.ndarray,
    *,
    variance: np.ndarray,
    poly_degree: int,
    derivative_order: int = 1,
    include_bias: bool = False,
    K: int | None = None,
    H_xt: float | None = None,
    p: int | None = None,
    weak_seed: int | None = None,
) -> IntraTrajectoryGLSData:
    """Construct weak / weighted-weak libraries for one ODE trajectory segment."""

    data = np.asarray(data, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    variance = np.asarray(variance, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must have shape (n_steps, n_states).")
    if variance.shape != (data.shape[0],):
        raise ValueError(
            f"variance must have shape ({data.shape[0]},), got {variance.shape}."
        )
    if t_grid.shape != (data.shape[0],):
        raise ValueError(f"t_grid must have shape ({data.shape[0]},), got {t_grid.shape}.")

    XT = t_grid[:, None]
    base_library = ps.PolynomialLibrary(
        degree=poly_degree,
        include_bias=include_bias,
    )

    if weak_seed is not None:
        np.random.seed(int(weak_seed))

    common_kwargs = {
        "function_library": base_library,
        "derivative_order": derivative_order,
        "spatiotemporal_grid": XT,
        "is_uniform": True,
        "include_bias": include_bias,
    }
    if K is not None:
        common_kwargs["K"] = K
    if H_xt is not None:
        common_kwargs["H_xt"] = H_xt
    if p is not None:
        common_kwargs["p"] = p

    weak_lib = DedupedWeakPDELibrary(**common_kwargs)

    if weak_seed is not None:
        np.random.seed(int(weak_seed))
    weighted_weak_lib_var = WeightedWeakPDELibrary(
        spatiotemporal_weights=variance,
        **common_kwargs,
    )

    if weak_seed is not None:
        np.random.seed(int(weak_seed))
    weighted_weak_lib_ones = WeightedWeakPDELibrary(
        spatiotemporal_weights=np.ones_like(variance),
        **common_kwargs,
    )

    return IntraTrajectoryGLSData(
        data=data,
        t_argument=t_grid,
        libraries={
            "No weighting": weak_lib,
            "Variance GLS": weighted_weak_lib_var,
            "Ones GLS": weighted_weak_lib_ones,
        },
        true_coefficients=np.empty((0, 0)),
    )


def evaluate_rollout_models(
    models: Mapping[str, PolynomialRolloutModel],
    validation_trajectories: Mapping[str, Sequence[np.ndarray]] | Sequence[np.ndarray],
    *,
    t_grid: np.ndarray,
    metric_name: str = "rollout_r2",
    source_col: str = "model",
) -> pd.DataFrame:
    """Evaluate rollout R^2 for a collection of models on held-out trajectories."""

    if isinstance(validation_trajectories, Mapping):
        grouped_trajectories = validation_trajectories.items()
    else:
        grouped_trajectories = [("validation", validation_trajectories)]

    rows: list[dict[str, Any]] = []
    for group_name, trajectories in grouped_trajectories:
        for traj_idx, trajectory in enumerate(trajectories):
            trajectory = np.asarray(trajectory, dtype=float)
            for model_name, model in models.items():
                try:
                    pred = model.simulate(trajectory[0], t_grid)
                    score = rollout_r2_score(trajectory, pred)
                except Exception:
                    score = -1e12
                rows.append(
                    {
                        source_col: model_name,
                        "group": group_name,
                        "trajectory": traj_idx,
                        "metric": metric_name,
                        "value": float(score),
                    }
                )

    return pd.DataFrame(rows)


def evaluate_windowed_rollout_models(
    models: Mapping[str, PolynomialRolloutModel],
    validation_trajectories: Mapping[str, Sequence[np.ndarray]] | Sequence[np.ndarray] | np.ndarray,
    *,
    t_grid: np.ndarray,
    window_size: int,
    stride: int = 1,
    metric_name: str = "short_rollout_r2",
    source_col: str = "model",
) -> pd.DataFrame:
    """Evaluate rollout R^2 on overlapping short windows from held-out trajectories."""

    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.ndim != 1 or t_grid.size < 2:
        raise ValueError("t_grid must be a 1D array with at least two time samples.")
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    if isinstance(validation_trajectories, Mapping):
        grouped_trajectories = validation_trajectories.items()
    elif isinstance(validation_trajectories, np.ndarray):
        grouped_trajectories = [("validation", [validation_trajectories])]
    else:
        grouped_trajectories = [("validation", validation_trajectories)]

    rows: list[dict[str, Any]] = []
    for group_name, trajectories in grouped_trajectories:
        for traj_idx, trajectory in enumerate(trajectories):
            trajectory = np.asarray(trajectory, dtype=float)
            if trajectory.ndim != 2:
                raise ValueError("Each validation trajectory must have shape (n_steps, n_states).")
            if trajectory.shape[0] < 2:
                raise ValueError("Each validation trajectory must contain at least two samples.")
            if t_grid.shape[0] < trajectory.shape[0]:
                raise ValueError("t_grid must be at least as long as each validation trajectory.")

            local_t = t_grid[: trajectory.shape[0]]
            local_window_size = min(int(window_size), trajectory.shape[0])
            if trajectory.shape[0] <= local_window_size:
                start_indices = [0]
            else:
                start_indices = list(
                    range(0, trajectory.shape[0] - local_window_size + 1, int(stride))
                )
                final_start = trajectory.shape[0] - local_window_size
                if start_indices[-1] != final_start:
                    start_indices.append(final_start)

            for model_name, model in models.items():
                for window_idx, start in enumerate(start_indices):
                    stop = start + local_window_size
                    window_true = trajectory[start:stop]
                    window_t = local_t[start:stop] - local_t[start]
                    try:
                        window_pred = model.simulate(window_true[0], window_t)
                        score = rollout_r2_score(window_true, window_pred)
                    except Exception:
                        score = -1e12
                    rows.append(
                        {
                            source_col: model_name,
                            "group": group_name,
                            "trajectory": traj_idx,
                            "window": window_idx,
                            "start_index": start,
                            "metric": metric_name,
                            "value": float(score),
                        }
                    )

    return pd.DataFrame(rows)


def format_model_equations(
    models: Mapping[str, PolynomialRolloutModel],
    *,
    precision: int = 3,
    tol: float = 1e-10,
) -> str:
    """Return a compact printable block of equations for several models."""

    sections: list[str] = []
    for model_name, model in models.items():
        lines = [model_name]
        lines.extend(model.equations(precision=precision, tol=tol))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


__all__ = [
    "PolynomialRolloutModel",
    "build_polynomial_intra_trajectory_artifacts",
    "build_polynomial_rollout_models",
    "evaluate_rollout_models",
    "evaluate_windowed_rollout_models",
    "format_model_equations",
    "rollout_r2_score",
    "split_single_trajectory",
    "split_trajectory_list",
]
