"""Does the weighting's advantage depend on the coverage it is measured at?

Coverage sets how much the test functions overlap, and overlap is what makes the
weak covariance ill-conditioned: at fixed support, cond(Sigma) runs from 1.4e3 at
coverage 1 to 6.4e6 at coverage 20, with 47 duplicate domains at the top. A
whitener built on a near-singular Sigma divides round-off directions by the
nugget, which moves the solution a great deal -- but for a reason that has
nothing to do with the noise model.

Part II used to run near coverage 6-10, so the question was whether the old
design's gaps were mostly that conditioning rather than the weighting.

The answer is that it goes the other way, and this script exists to record it.
Raising coverage does not flatter the full GLS rungs -- it wrecks them. On
Lorenz, Variance GLS goes from level with the unweighted rung at coverage 2 to
4.9x worse at coverage 20, and Ones GLS to 2.9x worse, while the diagonal rung
holds between 0.88 and 1.07 throughout. Both facts follow from the same place:
overlap is what puts mass off Sigma's diagonal, cond(Sigma) runs 8e3 to 6.4e6
over that range, and a diagonal matrix cannot be made ill-conditioned by overlap
at all. So the off-diagonal part of the covariance model is the fragile part,
and the diagonal part is not.

Everything else is held fixed: same support, same threshold, same seeds.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from mfsindy.cases.lorenz import (LorenzIntraTrajectoryGLSConfig,
                                  run_lorenz_intra_trajectory_gls_experiment)
from mfsindy.cases.hopf import (HopfIntraTrajectoryGLSConfig,
                                run_hopf_intra_trajectory_gls_experiment)

CASES = {
    "lorenz": (LorenzIntraTrajectoryGLSConfig, run_lorenz_intra_trajectory_gls_experiment, 0.1, 0.2),
    "hopf": (HopfIntraTrajectoryGLSConfig, run_hopf_intra_trajectory_gls_experiment, 0.5, 0.2),
}
METHODS = ["No weighting", "Ones GLS", "Variance diag", "Variance GLS"]
COVERAGES = (1, 2, 5, 10, 20)
N = 30

rows = []
for case, (Cfg, runner, H, thr) in CASES.items():
    probe = Cfg()
    extent = probe.t1 - probe.t0
    for coverage in COVERAGES:
        K = max(2, int(round(coverage * extent / (2 * H))))
        cfg = Cfg(n_runs=N, H_xt=H, K=K, stlsq_threshold=thr,
                  results_dir="/tmp", results_filename=f"cov_{case}.csv")
        cfg.methods = METHODS
        errors, _, _ = runner(cfg)
        l1 = errors[errors.metric == "L1"]
        med = {m: float(l1[l1.method == m]["value"].median()) for m in METHODS}
        rows.append(dict(case=case, coverage=coverage, K=K, **med))
        print(f"  {case} coverage {coverage:2d} (K={K}) done", flush=True)

res = pd.DataFrame(rows)
print("\n\nMedian L1 per rung, and each weighted rung relative to identity\n")
for case, g in res.groupby("case", sort=False):
    print(f"--- {case} (support and threshold fixed; only coverage varies)")
    show = g.set_index("coverage")[["K"] + METHODS].copy()
    for m in METHODS[1:]:
        show[f"{m} /I"] = (g.set_index("coverage")[m] / g.set_index("coverage")["No weighting"])
    print(show.to_string(float_format=lambda v: f"{v:.4g}"))
    print()
