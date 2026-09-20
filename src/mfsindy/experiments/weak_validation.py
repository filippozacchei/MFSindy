"""Held-out weak-form validation helpers for PDE-style examples."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeakValidationBlock:
    """One held-out weak-form block used for candidate scoring."""

    theta: np.ndarray
    rhs: np.ndarray
    group: str = "validation"
    trajectory: int = 0
    block: int = 0


def weak_r2_score(
    y_true: np.ndarray, y_pred: np.ndarray, *, per_state: bool = True
) -> float:
    """R^2 for weak-form targets.

    ``per_state`` scores each state component on its own scale and averages,
    rather than pooling every component into one sum of squares. Pooling lets
    whichever component has the largest weak target decide the score outright:
    the isothermal flow carries (u, v, rho) together, so a pooled score would
    tune the library to the largest of the three and leave the others
    unweighted. Averaging normalised scores asks each component to be explained
    as well as its own variance allows.

    A component that is constant over the block carries no information to
    explain and is skipped rather than scored -1e12, which would otherwise let
    one flat component condemn an entire candidate.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}.")
    if not np.all(np.isfinite(y_pred)):
        return -1e12

    residual = y_true - y_pred
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)

    if not per_state or y_true.ndim < 2 or y_true.shape[1] < 2:
        denom = float(np.sum(centered**2))
        if denom <= 0.0:
            return -1e12
        return float(1.0 - np.sum(residual**2) / denom)

    scores: list[float] = []
    for column in range(y_true.shape[1]):
        denom = float(np.sum(centered[:, column] ** 2))
        if denom <= 0.0:
            continue
        scores.append(1.0 - float(np.sum(residual[:, column] ** 2)) / denom)
    if not scores:
        return -1e12
    return float(np.mean(scores))


def evaluate_weak_form_models(
    coefficient_map: Mapping[str, np.ndarray],
    validation_blocks: Sequence[WeakValidationBlock],
    *,
    metric_name: str = "weak_r2",
    source_col: str = "model",
    per_state: bool = True,
) -> pd.DataFrame:
    """Score coefficient maps on held-out weak-form blocks."""

    rows: list[dict[str, Any]] = []
    for block in validation_blocks:
        theta = np.asarray(block.theta, dtype=float)
        rhs = np.asarray(block.rhs, dtype=float)
        for model_name, coefficients in coefficient_map.items():
            coef = np.asarray(coefficients, dtype=float)
            try:
                pred = theta @ coef.T
                score = weak_r2_score(rhs, pred, per_state=per_state)
            except Exception:
                score = -1e12
            rows.append(
                {
                    source_col: model_name,
                    "group": block.group,
                    "trajectory": block.trajectory,
                    "block": block.block,
                    "metric": metric_name,
                    "value": float(score),
                }
            )
    return pd.DataFrame(rows)


def split_spatiotemporal_trajectory(
    trajectory: np.ndarray,
    t_grid: np.ndarray,
    *,
    time_axis: int | None = None,
    validation_fraction: float = 0.2,
    overlap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a spatiotemporal trajectory along its time axis."""

    trajectory = np.asarray(trajectory)
    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be one-dimensional.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1).")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    if time_axis is None:
        candidates = [axis for axis, size in enumerate(trajectory.shape) if size == t_grid.shape[0]]
        if not candidates:
            raise ValueError(
                "Could not infer the time axis: no trajectory dimension matches len(t_grid)."
            )
        time_axis = candidates[0]

    time_axis = int(time_axis)
    if time_axis < 0:
        time_axis += trajectory.ndim
    if not 0 <= time_axis < trajectory.ndim:
        raise ValueError(f"Invalid time_axis {time_axis} for shape {trajectory.shape}.")
    if trajectory.shape[time_axis] != t_grid.shape[0]:
        raise ValueError("trajectory and t_grid must agree along the time axis.")

    n_total = trajectory.shape[time_axis]
    n_val = max(2, int(np.ceil(n_total * validation_fraction)))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val
    val_start = max(0, n_train - overlap)

    train_slices = [slice(None)] * trajectory.ndim
    val_slices = [slice(None)] * trajectory.ndim
    train_slices[time_axis] = slice(0, n_train)
    val_slices[time_axis] = slice(val_start, None)

    train = trajectory[tuple(train_slices)].copy()
    val = trajectory[tuple(val_slices)].copy()
    train_t = t_grid[:n_train] - t_grid[0]
    val_t = t_grid[val_start:] - t_grid[val_start]
    return train, val, train_t, val_t


def get_library_feature_names(
    library: Any,
    reference_data: np.ndarray,
    *,
    input_features: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Best-effort feature-name extraction for a fitted SINDy library."""

    try:
        library.fit([reference_data])
    except Exception:
        try:
            library.fit(reference_data)
        except Exception:
            return ()

    getter = getattr(library, "get_feature_names", None)
    if getter is None:
        return ()

    for kwargs in (
        {"input_features": list(input_features)} if input_features is not None else None,
        {},
    ):
        if kwargs is None:
            continue
        try:
            names = getter(**kwargs)
            return tuple(str(name) for name in names)
        except Exception:
            continue

    try:
        names = getter(list(input_features)) if input_features is not None else getter()
        return tuple(str(name) for name in names)
    except Exception:
        return ()


def format_coefficient_equations(
    coefficient_map: Mapping[str, np.ndarray],
    *,
    feature_names: Sequence[str],
    state_names: Sequence[str],
    precision: int = 3,
    tol: float = 1e-10,
) -> str:
    """Format a map of coefficient matrices as readable equations."""

    feature_names = tuple(feature_names)
    state_names = tuple(state_names)
    sections: list[str] = []
    for model_name, coefficients in coefficient_map.items():
        coef = np.asarray(coefficients, dtype=float)
        if coef.ndim == 1:
            coef = coef[None, :]
        local_features = feature_names or tuple(f"term_{j}" for j in range(coef.shape[1]))
        lines = [model_name]
        for state_name, row in zip(state_names, coef, strict=True):
            terms: list[tuple[str, str]] = []
            for value, feature_name in zip(row, local_features, strict=True):
                value = float(value)
                if abs(value) <= tol:
                    continue
                sign = "-" if value < 0 else "+"
                magnitude = abs(value)
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
            lines.append(f"{state_name} = {rhs}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)



# ---------------------------------------------------------------------------
# Choosing the validation weak system
# ---------------------------------------------------------------------------


def weak_target_noise_ceiling(
    library: Any,
    rhs: np.ndarray,
    *,
    sigma: float,
) -> float:
    r"""Highest weak R^2 any model can reach against this validation target.

    The validation target is ``b_k = \int \dot\phi_k u``, so observation noise
    enters it with variance ``sigma^2 ||\dot\phi_k||^2`` per row. That noise is
    irreducible: it caps the score of the true coefficients themselves. Writing
    the cap out,

        ceiling = 1 - n_states * sigma^2 * sum_k ||\dot\phi_k||^2 / SS_tot(b),

    every term of which is available from the assembled system plus the known
    noise level, so the cap is computable without the true coefficients. It
    matches the measured ceiling to three decimals on Lorenz.

    The cap is what rules out narrow validation supports: ``||\dot\phi||^2``
    grows as the support shrinks, which is the same noise amplification that
    makes a finite-difference derivative useless, only milder.
    """

    rhs = np.asarray(rhs, dtype=float)
    noise_power = 0.0
    for k in range(int(library.K)):
        weights = np.asarray(library.fulltweights[k], dtype=float).ravel()
        noise_power += float(np.sum(weights**2))

    centered = rhs - np.mean(rhs, axis=0, keepdims=True)
    if rhs.ndim < 2 or rhs.shape[1] < 2:
        ss_total = float(np.sum(centered**2))
        if ss_total <= 0.0:
            return -np.inf
        return float(1.0 - float(sigma) ** 2 * noise_power / ss_total)

    # weak_r2_score averages per-state scores, so the ceiling has to be the
    # average of the per-state ceilings: the noise power per state is the same
    # but each is measured against that state's own variance.
    ceilings: list[float] = []
    for column in range(rhs.shape[1]):
        ss_total = float(np.sum(centered[:, column] ** 2))
        if ss_total <= 0.0:
            continue
        ceilings.append(1.0 - float(sigma) ** 2 * noise_power / ss_total)
    if not ceilings:
        return -np.inf
    return float(np.mean(ceilings))


@dataclass(frozen=True)
class ValidationSupport:
    """The validation weak system a benchmark scores its candidates against."""

    H_val: Any
    ceiling: float
    kappa: float
    K: int
    table: pd.DataFrame
    satisfied: bool = True

    def summary(self) -> str:
        status = "" if self.satisfied else "  (no candidate met both bounds)"
        return (
            f"H_val={self.H_val}  K={self.K}  ceiling={self.ceiling:.4f}  "
            f"kappa={self.kappa:.4f}{status}"
        )


def select_validation_support(
    build: Callable[[Any], tuple[Any, np.ndarray]],
    *,
    candidates: Sequence[Any],
    sigma: float,
    kappa_fn: Callable[[Any, Any], float] | None = None,
    min_ceiling: float = 0.99,
    max_kappa: float = float("inf"),
    min_rows: int | None = None,
) -> ValidationSupport:
    """Pick the support width of the held-out weak system, by a stated rule.

    Every candidate in the hyperparameter grid is scored against one fixed
    validation system, so that system's support is a choice in its own right and
    is made here rather than inherited from whichever candidate is being scored.

    What binds the choice is ``min_ceiling``: it rejects supports so narrow that
    noise in ``b`` caps the attainable score, since the weak target inherits
    observation noise through ``||phi_dot||`` exactly as a derivative estimate
    would -- see :func:`weak_target_noise_ceiling`. Among the candidates that
    clear it the **smallest** is taken, which yields the most rows, and rows are
    what give the metric resolution between candidates.

    ``max_kappa`` defaults to no bound, and should normally be left that way.
    Kappa governs the *covariance model* -- the approximation that drops the
    feature-driven term from Cov(R) -- so it is a property of the rungs that
    whiten by that covariance, not of this system: the validation library is
    built unweighted and scored with an unweighted R^2, so no covariance model
    is invoked here at all. Kappa is still computed and reported in ``table``
    when ``kappa_fn`` is given, as context rather than as a constraint.

    ``build(H)`` returns ``(library, rhs)`` for one candidate width;
    ``kappa_fn(library, H)`` returns that width's validity ratio, or None to
    skip the diagnostic entirely.
    """

    rows: list[dict[str, Any]] = []
    built: dict[int, tuple[Any, float, float, int]] = {}
    for idx, H in enumerate(candidates):
        library, rhs = build(H)
        ceiling = weak_target_noise_ceiling(library, rhs, sigma=sigma)
        kappa = float("nan") if kappa_fn is None else float(kappa_fn(library, H))
        K = int(library.K)
        built[idx] = (H, ceiling, kappa, K)
        rows.append(
            {
                "H_val": H,
                "K": K,
                "ceiling": ceiling,
                "kappa": kappa,
                "ceiling_ok": bool(ceiling >= min_ceiling),
                "kappa_ok": bool(np.isnan(kappa) or kappa <= max_kappa),
                "rows_ok": bool(min_rows is None or K >= int(min_rows)),
            }
        )
    table = pd.DataFrame(rows)
    # A column that is True for every row states nothing, and sitting beside a
    # real bound it reads as a check that was made and passed. Drop the ones
    # whose bound was never set.
    if not np.isfinite(max_kappa):
        table = table.drop(columns=["kappa_ok"])
    if min_rows is None:
        table = table.drop(columns=["rows_ok"])
    if kappa_fn is None:
        table = table.drop(columns=["kappa"])

    mask = table["ceiling_ok"].to_numpy(dtype=bool)
    if "kappa_ok" in table:
        mask &= table["kappa_ok"].to_numpy(dtype=bool)
    if "rows_ok" in table:
        mask &= table["rows_ok"].to_numpy(dtype=bool)
    passing = table.index[mask]
    satisfied = len(passing) > 0
    if satisfied:
        # Smallest qualifying width: most rows, hence the finest discrimination.
        order = sorted(passing, key=lambda i: built[i][3], reverse=True)
        chosen = order[0]
    else:
        # Nothing clears both bounds. Prefer a trustworthy target over a
        # sensitive one: take the widest support still under max_kappa, else
        # the highest ceiling outright, and say so rather than failing quietly.
        fallback = (
            table.index[table["kappa_ok"]] if "kappa_ok" in table else table.index
        )
        pool = list(fallback) if len(fallback) else list(table.index)
        chosen = max(pool, key=lambda i: built[i][1])
        warnings.warn(
            f"No validation support reached ceiling >= {min_ceiling} "
            f"(max_kappa={max_kappa}); falling back to "
            f"H_val={built[chosen][0]} (ceiling={built[chosen][1]:.4f}, "
            f"kappa={built[chosen][2]:.4f}). Widen the candidate list.",
            RuntimeWarning,
            stacklevel=2,
        )

    H, ceiling, kappa, K = built[chosen]
    return ValidationSupport(
        H_val=H, ceiling=ceiling, kappa=kappa, K=K, table=table, satisfied=satisfied
    )


__all__ = [
    "WeakValidationBlock",
    "evaluate_weak_form_models",
    "format_coefficient_equations",
    "get_library_feature_names",
    "ValidationSupport",
    "select_validation_support",
    "split_spatiotemporal_trajectory",
    "weak_r2_score",
    "weak_target_noise_ceiling",
]
