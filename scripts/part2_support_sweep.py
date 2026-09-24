"""Does kappa predict where the covariance weighting helps?

The Part II search picked wide supports in every case, which on Lorenz means
kappa > 1 -- outside the region where the covariance model is derived. If the
weighting's advantage tracks kappa, then Lorenz's flat result is the model
being used outside its stated validity region rather than the model failing,
and the two are distinguishable by sweeping the support.

Threshold is swept alongside and the best median over thresholds is taken per
(support, rung), so what is compared at each support is what each rung could
achieve there -- otherwise a support would be judged at a threshold that suits
one rung and not another.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

from mfsindy.cases.lorenz import (LorenzIntraTrajectoryGLSConfig,
                                  run_lorenz_intra_trajectory_gls_experiment)
from mfsindy.cases.hopf import (HopfIntraTrajectoryGLSConfig,
                                run_hopf_intra_trajectory_gls_experiment)
from mfsindy.cases.pendulum import (PendulumIntraTrajectoryGLSConfig,
                                    run_pendulum_intra_trajectory_gls_experiment)

CASES = {
    "lorenz": (LorenzIntraTrajectoryGLSConfig, run_lorenz_intra_trajectory_gls_experiment,
               [0.05, 0.1, 0.2, 0.5, 1.0]),
    "hopf": (HopfIntraTrajectoryGLSConfig, run_hopf_intra_trajectory_gls_experiment,
             [0.02, 0.05, 0.1, 0.2, 0.5]),
    "pendulum": (PendulumIntraTrajectoryGLSConfig, run_pendulum_intra_trajectory_gls_experiment,
                 [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]),
}
SUPPORTS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
METHODS = ["No weighting", "Ones GLS", "Variance diag", "Variance GLS"]
N_RUNS = 25
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)

rows = []
t_start = time.time()
for case, (Cfg, runner, thresholds) in CASES.items():
    for H in SUPPORTS:
        for thr in thresholds:
            cfg = Cfg(n_runs=N_RUNS, H_xt=H, K=None, stlsq_threshold=thr,
                      results_dir=str(out), results_filename=f"sweep_{case}.csv")
            cfg.methods = METHODS
            try:
                errors, *_ = runner(cfg)
            except Exception as exc:
                print(f"  {case} H={H} thr={thr} FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            l1 = errors[errors.metric == "L1"]
            for m in METHODS:
                v = l1[l1.method == m]["value"]
                rows.append(dict(case=case, H_xt=H, thr=thr, method=m,
                                 median=float(v.median()), mean=float(v.mean())))
        print(f"  {case} H={H} done ({time.time()-t_start:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(out / "kappa_sweep.csv", index=False)

print("\n\nBest median L1 over thresholds, per support and rung"
      "  (ratio = rung / No weighting; <1 means the weighting helps)\n")
for case, g in res.groupby("case", sort=False):
    print(f"--- {case}")
    best = g.groupby(["H_xt", "method"])["median"].min().unstack()[METHODS]
    ratio = best.div(best["No weighting"], axis=0).drop(columns="No weighting")
    table = pd.concat([best.round(4), ratio.round(3).add_suffix(" /base")], axis=1)
    print(table.to_string())
    print()
