"""Sensitivity of the comparison to the fixed design choices (repo only, not the paper).

The support and the threshold are selected from the data; three choices are fixed
in advance. Each is varied here, one at a time, around the protocol:

  overlap      test functions per point on average: 1, 2 (protocol), 4
  order        test-function polynomial degree p: 2 (protocol), 16
  grid         candidate supports: trimmed T/200..T/20 (protocol), wide to T/5

For each case, part and variant it runs the Monte Carlo and reports the paired
median error ratio of the proposed weighting against each other rung, and the
share of runs it wins. The question is whether any reasonable alternative flips
the ordering, not which variant is best.

    python scripts/sensitivity.py [--cases lorenz pendulum] [--runs 20] [--jobs 4]
"""

from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

#: Part I compares WMF (MF_w) with these, Part II Variance GLS with these.
REFERENCE = {1: ("MF_w", ["HF", "MF", "PMF", "VMF"]),
             2: ("Variance GLS", ["No weighting", "Ones GLS", "Variance diag"])}
WIDE_DIVISORS = (200.0, 100.0, 50.0, 20.0, 10.0, 5.0)


def with_overlap(cfg, overlap: float, extent: float):
    """``cfg`` as an instance of a subclass that derives K from ``overlap``.

    The runner fits each candidate support as ``replace(cfg, H_xt=H, K=None)``;
    ``replace`` runs ``__post_init__``, so K is set there from the support. The
    case builders take an explicit K as given. Defined here, the subclass is
    pickled with the config into the parallel workers.
    """

    @dataclass
    class Overlap(type(cfg)):
        def __post_init__(self):
            if self.K is None and self.H_xt is not None:
                self.K = max(2, int(round(overlap / 2.0 * extent / float(self.H_xt))))

    return Overlap(**{f.name: getattr(cfg, f.name) for f in fields(cfg)})


def variants(part: int, main):
    """(name, cfg factory) pairs: the protocol, then one change at a time."""

    extent = (lambda c: c.T_train) if part == 1 else (lambda c: c.t1 - c.t0)

    def wide(**kw):
        cfg = main.make_config(**kw)
        cfg.H_grid = [extent(cfg) / d for d in WIDE_DIVISORS]
        return cfg

    def overlap(value):
        def build(**kw):
            cfg = main.make_config(**kw)
            return with_overlap(cfg, value, extent(cfg))
        return build

    return [("protocol", main.make_config),
            ("overlap 1", overlap(1.0)),
            ("overlap 4", overlap(4.0)),
            ("order p=16", lambda **kw: main.make_config(p=16, **kw)),
            ("wide grid", wide)]


def ratios(errors: pd.DataFrame, part: int) -> dict:
    col = "model" if part == 1 else "method"
    metric = errors.metric.unique()[0]               # MAE (Part I) or L1 (Part II)
    e = errors[errors.metric == metric].pivot(index="run", columns=col, values="value")
    ref, others = REFERENCE[part]
    out = {"ref median": float(e[ref].median())}
    for m in others:
        out[f"/{m}"] = f"{np.median(e[ref] / e[m]):.2f} ({100 * np.mean(e[ref] < e[m]):.0f}%)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["lorenz", "pendulum"])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "scratch" / "sensitivity"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in args.cases:
        for part in (1, 2):
            sys.path.insert(0, str(ROOT / "examples" / case))
            main_mod = importlib.import_module(f"main{part}")
            for name, factory in variants(part, main_mod):
                tag = f"{case}_part{part}_{name.replace(' ', '_').replace('=', '')}"
                cfg = factory(n_runs=args.runs, n_jobs=args.jobs, results_dir=str(out),
                              results_filename=f"{tag}.csv")
                errors = main_mod.run_monte_carlo(cfg, verbose=False)
                rows.append({"case": case, "part": part, "variant": name, **ratios(errors, part)})
                print(f"{case} part {part} {name}: done", flush=True)
            sys.path.pop(0)
            del sys.modules[f"main{part}"]

    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    for (case, part), g in table.groupby(["case", "part"]):
        ref = REFERENCE[part][0]
        print(f"\n{case}, Part {'I' if part == 1 else 'II'}: {ref} error / rung error, "
              "paired median (share of runs it wins)")
        print(g.drop(columns=["case", "part"]).dropna(axis=1, how="all").to_string(index=False))


if __name__ == "__main__":
    main()
