"""Each rung at its own best design, 100 seeds, paired against the unweighted one.

The committed Monte Carlo runs each rung where the search put it. The search
cannot separate its top candidates -- in all twelve rung/case combinations the
winner leads the runner-up by less than one standard error -- so the tie is
broken by the one-sem parsimony rule, and that rule lands different rungs in
different places. This asks the other question: with every rung at the design
that is actually best for it, how much is the weighting worth?

The designs come from the committed support sweep, so they are chosen by
coefficient error and therefore oracle. That is the point: this is an upper
bound on what the weighting can deliver, to sit beside the search-tuned number,
which is what a practitioner gets. Neither alone is the answer.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

from mfsindy.cases.lorenz import (LorenzIntraTrajectoryGLSConfig,
                                  run_lorenz_intra_trajectory_gls_experiment)
from mfsindy.cases.hopf import (HopfIntraTrajectoryGLSConfig,
                                run_hopf_intra_trajectory_gls_experiment)
from mfsindy.cases.pendulum import (PendulumIntraTrajectoryGLSConfig,
                                    run_pendulum_intra_trajectory_gls_experiment)

CASES = {
    "lorenz": (LorenzIntraTrajectoryGLSConfig, run_lorenz_intra_trajectory_gls_experiment),
    "hopf": (HopfIntraTrajectoryGLSConfig, run_hopf_intra_trajectory_gls_experiment),
    "pendulum": (PendulumIntraTrajectoryGLSConfig, run_pendulum_intra_trajectory_gls_experiment),
}
METHODS = ["No weighting", "Ones GLS", "Variance diag", "Variance GLS"]
ROOT = Path("/Users/filippozacchei/COde/MFSindy")
sweep = pd.read_csv(ROOT / "results" / "part2_support_sweep.csv")

rows = []
for case, (Cfg, runner) in CASES.items():
    g = sweep[sweep.case == case]
    series = {}
    for rung in METHODS:
        best = g[g.method == rung].loc[g[g.method == rung]["median"].idxmin()]
        cfg = Cfg(n_runs=100, stlsq_threshold=float(best.thr), H_xt=float(best.H_xt),
                  K=None, results_dir="/tmp", results_filename=f"erb_{case}.csv")
        cfg.methods = [rung]
        errors, _, _ = runner(cfg)
        l1 = errors[(errors.metric == "L1") & (errors.method == rung)]
        series[rung] = l1.set_index("run")["value"].sort_index()
        print(f"{case:<9} {rung:<15} @ (thr={best.thr}, H={best.H_xt}) "
              f"median L1 {series[rung].median():.5f}", flush=True)
    base = series["No weighting"]
    for rung in METHODS[1:]:
        a, b = base.align(series[rung], join="inner")
        rows.append(dict(case=case, rung=rung,
                         ratio=float(b.median() / a.median()),
                         wins=int((b < a).sum()), n=len(a),
                         p=float(wilcoxon(a, b).pvalue)))
    print(flush=True)

print("\nEach rung at its own best design, paired against unweighted at its own best:\n")
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
res.to_csv(ROOT / "results" / "part2_matched_design.csv", index=False)
