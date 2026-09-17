"""Per-rung hyperparameter selection for the multi-fidelity experiments.

Every rung is tuned on its own grid, so no strategy runs on hyperparameters
chosen to suit another. The grid is evaluated once per point: the fit helpers
return all rungs together, so scoring each rung from the same evaluation costs
no more than tuning a single one.

Protocol
--------
* One grid per benchmark, shared by every rung, so the comparison is between
  rungs rather than between search spaces.
* One held-out validation set, identical for every rung. It is scored at the
  high-fidelity noise level, otherwise rungs fitted mostly to low-fidelity data
  would be judged against an easier target.
* The tuning draw uses a seed disjoint from the evaluation seeds, so reported
  errors are not in-sample with respect to selection.
* A rung whose selection sits on an edge of the grid is flagged: the grid was
  too narrow for it, and the comparison is unfair to that rung until widened.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import pandas as pd
from tqdm import tqdm

from .hyperparameters import _clone_with_updates, _coerce_results_frame

#: Seed offset for the tuning draw, kept away from the evaluation seeds.
TUNING_SEED_OFFSET = 900_000


@dataclass
class RungTuning:
    """Outcome of the grid search for one rung."""

    rung: str
    best_params: Dict[str, Any]
    best_score: float
    at_boundary: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "at_grid_boundary": self.at_boundary,
        }


def _boundary_axes(params: Mapping[str, Any], grid: Mapping[str, Sequence[Any]]) -> Dict[str, str]:
    """Axes whose selection sits on an edge, and which end, so the grid can be
    extended in the right direction."""

    flagged: Dict[str, str] = {}
    for name, values in grid.items():
        values = list(values)
        if len(values) < 3:
            # With fewer than three points every choice is an edge, so the flag
            # would carry no information.
            continue
        chosen = params.get(name)
        if chosen == values[0]:
            flagged[name] = "low"
        elif chosen == values[-1]:
            flagged[name] = "high"
    return flagged


def tune_rungs(
    base_config: Any,
    *,
    param_grid: Mapping[str, Sequence[Any]],
    evaluate: Callable[[Any], Any],
    rungs: Sequence[str],
    metric: str = "rollout_r2",
    source_col: str = "model",
    maximize: bool = True,
    reducer: Callable[[pd.Series], float] = lambda s: float(s.mean()),
    progress_desc: str = "Tuning grid",
) -> tuple[Dict[str, RungTuning], pd.DataFrame]:
    """Select hyperparameters separately for each rung from one pass over the grid.

    ``evaluate`` is called once per grid point with a cloned config and must
    return a long-format frame carrying one row per rung and metric, as produced
    by :func:`mfsindy.experiments.evaluate_rollout_models`.

    Returns the per-rung selections and the full score table, which belongs in
    the paper's reproducibility appendix.
    """

    if not param_grid:
        raise ValueError("param_grid must contain at least one hyperparameter.")

    names = list(param_grid)
    values = [list(param_grid[n]) for n in names]
    combos = list(itertools.product(*values))
    rows: list[dict[str, Any]] = []

    failures: list[tuple[dict, str]] = []
    disable_progress = os.environ.get("MFSINDY_DOCS_BUILD") == "1"
    bar = tqdm(combos, desc=progress_desc, disable=disable_progress)
    for combo in bar:
        updates = dict(zip(names, combo))
        bar.set_postfix({k: v for k, v in updates.items()}, refresh=False)
        candidate = _clone_with_updates(base_config, updates)
        try:
            frame = _coerce_results_frame(evaluate(candidate))
        except Exception as exc:  # noqa: BLE001 - a bad grid point must not end the search
            failures.append((updates, f"{type(exc).__name__}: {exc}"))
            for rung in rungs:
                rows.append({**updates, "rung": rung, "score": float("nan"),
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        for rung in rungs:
            mask = (frame[source_col] == rung) & (frame["metric"] == metric)
            score = reducer(frame.loc[mask, "value"]) if mask.any() else float("nan")
            rows.append({**updates, "rung": rung, "score": score, "error": None})

    bar.close()
    if failures:
        print(f"{len(failures)} of {len(combos)} grid points failed and were skipped:")
        for updates, message in failures[:5]:
            print(f"  {updates}: {message}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more; see the 'error' column")
    table = pd.DataFrame(rows)

    selections: Dict[str, RungTuning] = {}
    for rung in rungs:
        sub = table[table["rung"] == rung].dropna(subset=["score"])
        if sub.empty:
            raise ValueError(f"No scores recorded for rung {rung!r}.")
        best = sub.loc[sub["score"].idxmax() if maximize else sub["score"].idxmin()]
        params = {n: best[n] for n in names}
        selections[rung] = RungTuning(
            rung=rung,
            best_params=params,
            best_score=float(best["score"]),
            at_boundary=_boundary_axes(params, param_grid),
        )
    return selections, table


def run_with_tuned_configs(
    runner: Callable[[Any], Any],
    cfg_by_rung: Mapping[str, Any],
) -> pd.DataFrame:
    """Run the Monte Carlo once per rung, each with its own tuned config.

    Each run fits every rung, but only the rung the config was tuned for is
    kept, so no strategy is reported on another's hyperparameters. Returns the
    concatenated long-format error frame.
    """

    frames = []
    disable_progress = os.environ.get("MFSINDY_DOCS_BUILD") == "1"
    for rung, cfg in tqdm(
        list(cfg_by_rung.items()), desc="Rungs", disable=disable_progress
    ):
        result = runner(cfg)
        frame = _coerce_results_frame(result)
        kept = frame[frame["model"] == rung]
        if kept.empty:
            raise ValueError(f"Run for rung {rung!r} produced no rows for that rung.")
        frames.append(kept)
    return pd.concat(frames, ignore_index=True)


def save_tuning(
    path: str | Path,
    *,
    selections: Mapping[str, RungTuning],
    param_grid: Mapping[str, Sequence[Any]],
    table: pd.DataFrame | None = None,
    fixed: Mapping[str, Any] | None = None,
) -> Path:
    """Write the selection, the grid and the fixed settings to JSON.

    ``fixed`` should carry everything held constant but reportable: test-function
    family and order, ensemble size, subsample fraction, library degree, seeds.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "param_grid": {k: list(v) for k, v in param_grid.items()},
        "fixed": dict(fixed or {}),
        "selection": {r: s.as_dict() for r, s in selections.items()},
    }
    if table is not None:
        payload["scores"] = table.to_dict(orient="records")
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
