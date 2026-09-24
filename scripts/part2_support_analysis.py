"""What the Part II support sweep says, from the CSV it wrote.

Two questions the tuned Monte Carlo cannot answer on its own:

1. Does the weighting's benefit track kappa? The covariance model neglects a
   term whose size relative to the retained one is kappa, so the benefit should
   shrink as kappa grows. Tested per case as an ordering (Kendall), since the
   prediction is monotone rather than functional and there are seven supports.

2. Is each rung run where it is actually best? The search maximises held-out
   weak R^2 while the paper reports coefficient L1. Where those disagree a rung
   is run somewhere other than its own optimum, and a comparison between rungs
   measures partly that mismatch rather than the weighting.

Run ``python scripts/part2_support_sweep.py <out_dir>`` first; this reads the
CSV it writes, together with each case's committed tuning record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results" / "part2_support_sweep.csv"
METHODS = ["No weighting", "Ones GLS", "Variance diag", "Variance GLS"]
CASES = ("lorenz", "hopf", "pendulum")


def tuning(case: str) -> dict:
    path = ROOT / "examples" / case / "results" / f"{case}_part2_tuning.json"
    return json.loads(path.read_text())


def kappa_by_support(case: str) -> dict[float, float]:
    return {
        round(float(row["H_xt"]), 6): float(row["kappa_median"])
        for row in tuning(case)["kappa_table"]
    }


def benefit_against_kappa(sweep: pd.DataFrame) -> None:
    print("=" * 72)
    print("Does the benefit track kappa?")
    print("=" * 72)
    for case in CASES:
        kappa = kappa_by_support(case)
        best = (
            sweep[sweep.case == case]
            .groupby(["H_xt", "method"])["median"]
            .min()
            .unstack()
        )
        table = pd.DataFrame(
            [
                {
                    "H_xt": H,
                    "kappa": kappa.get(round(float(H), 6)),
                    "L1_unweighted": best.loc[H, "No weighting"],
                    "ratio_var": best.loc[H, "Variance GLS"] / best.loc[H, "No weighting"],
                    "ratio_diag": best.loc[H, "Variance diag"] / best.loc[H, "No weighting"],
                }
                for H in best.index
            ]
        ).dropna()

        print(f"\n--- {case}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        for column in ("ratio_var", "ratio_diag"):
            tau, p = kendalltau(table["kappa"], table[column])
            print(f"    {column:<11} vs kappa:  tau = {tau:+.3f}   p = {p:.3f}")

        cheapest = table.loc[table["L1_unweighted"].idxmin()]
        strongest = table.loc[table["ratio_var"].idxmin()]
        print(
            f"    largest benefit at H={strongest.H_xt:g} (ratio {strongest.ratio_var:.2f}),"
            f" where the unweighted error is {strongest.L1_unweighted:.4f}"
        )
        print(
            f"    lowest error   at H={cheapest.H_xt:g} ({cheapest.L1_unweighted:.4f}),"
            f" where the benefit is only {cheapest.ratio_var:.2f}"
        )


def tuned_against_achievable(sweep: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("Is each rung run where it is best? (cost = tuned L1 / best L1 on the grid)")
    print("=" * 72)
    print("The best is a minimum over many noisy medians, so it is optimistically")
    print("biased; a cost near 1 is not evidence of anything, a large one is worth")
    print("re-running at the two configs directly.")
    for case in CASES:
        selection = {
            rung: entry["best_params"] for rung, entry in tuning(case)["selection"].items()
        }
        case_rows = sweep[sweep.case == case]
        print(f"\n--- {case}")
        print(
            f"{'rung':<15}{'tuned (thr, H)':>18}{'L1 there':>11}"
            f"{'best (thr, H)':>18}{'best L1':>11}{'cost':>8}"
        )
        for rung in METHODS:
            rows = case_rows[case_rows.method == rung]
            threshold = selection[rung]["stlsq_threshold"]
            support = selection[rung]["H_xt"]
            at_tuned = rows[
                np.isclose(rows.thr, threshold) & np.isclose(rows.H_xt, support)
            ]["median"]
            if at_tuned.empty:
                print(f"{rung:<15}{'(off the sweep grid)':>18}")
                continue
            tuned_l1 = float(at_tuned.iloc[0])
            best = rows.loc[rows["median"].idxmin()]
            print(
                f"{rung:<15}{f'({threshold}, {support})':>18}{tuned_l1:11.5f}"
                f"{f'({best.thr}, {best.H_xt})':>18}{best['median']:11.5f}"
                f"{tuned_l1 / best['median']:7.2f}x"
            )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SWEEP
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/part2_support_sweep.py first.")
    sweep = pd.read_csv(path)
    benefit_against_kappa(sweep)
    tuned_against_achievable(sweep)


if __name__ == "__main__":
    main()
