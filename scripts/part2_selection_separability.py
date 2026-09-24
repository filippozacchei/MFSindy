"""Can the search's score actually separate the candidates it is choosing between?

The search reports a score and a spread over its takes. If the winner leads the
runner-up by less than that spread, the score is not ranking the two -- the
choice between them is made by the one-standard-error tie-break, on parsimony,
which is a different criterion from the one the paper reports.

That matters because the tie-break does not land every rung in the same place.
On pendulum it sends the unweighted rung to a design 3.0x worse in coefficient
L1 than another point on the same grid (100 seeds, 93/100, p = 5e-17) while
sending both variance rungs to within 1% of theirs, which inflates the apparent
benefit of the weighting. Reported per rung so that asymmetry is visible.

Reads the committed tuning records; no runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASES = ("lorenz", "hopf", "pendulum", "burgers", "isothermal_flow")
TUNING_NAME = {"isothermal_flow": "navierstokes"}


def record(case: str) -> dict | None:
    stem = TUNING_NAME.get(case, case)
    path = ROOT / "examples" / case / "results" / f"{stem}_part2_tuning.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    for case in CASES:
        payload = record(case)
        if payload is None:
            print(f"\n======== {case}: no tuning record yet ========")
            continue

        scores = pd.DataFrame(payload["scores"])
        selection = payload["selection"]
        print(f"\n======== {case} ========")
        for rung, entry in selection.items():
            frame = scores[scores.rung == rung].sort_values("score", ascending=False)
            if len(frame) < 2:
                continue
            winner, runner_up = frame.iloc[0], frame.iloc[1]
            gap = float(winner.score - runner_up.score)
            sem = float(winner.score_sem)
            within = frame[frame.score >= winner.score - sem]
            verdict = "separated" if gap >= sem else f"tied ({len(within)} within 1 sem)"
            print(
                f"  {rung:<15} top score {winner.score:.4f} +- {sem:.4f},"
                f" gap to runner-up {gap:.4f}   {verdict}"
            )
            print(f"  {'':<15} selected {entry['best_params']}")


if __name__ == "__main__":
    main()
