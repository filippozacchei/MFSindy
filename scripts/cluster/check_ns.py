"""Checks before an isothermal-flow Monte Carlo at a new grid.

    python scripts/cluster/check_ns.py residual N        # clean-data residual at the truth
    python scripts/cluster/check_ns.py memory N PART OUT  # peak memory of one run, one process

``residual`` solves two flows and reports, per support, how far the analytic
coefficients are from fitting the clean weak system (the misfit the HF noise has
to dominate for the true model to be recoverable) and the smallest true
coefficient plain least squares on the true support returns (1 when exact).
``memory`` runs one Monte Carlo run of Part PART in one process, with 5 bags
(the bags barely change memory), and reports its peak: what the job script
needs as the GB per worker.
"""

import resource
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "isothermal_flow"))

import mfsindy.cases.isothermal_flow as F  # noqa: E402
from mfsindy.weighted_weak_pde_library import weak_system  # noqa: E402


def peak_gb() -> float:
    scale = 1e9 if sys.platform == "darwin" else 1e6  # ru_maxrss: bytes on macOS, kB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def residual(N: int) -> None:
    import main1

    cfg = main1.make_config(N=N)
    for seed in range(2):
        t0 = time.time()
        U, _, grid = F.generate_isothermal_ns_dataset(
            N=N, Nt=cfg.Nt, L=cfg.L, T=cfg.T, mu=cfg.mu, RT=cfg.RT, seed=seed
        )
        grid = np.asarray(grid, float)
        names = list(F.get_ns_isothermal_feature_names(cfg, grid=grid, reference_trajectory=U))
        C = F.build_true_ns_isothermal_coefficients(names, RT=cfg.RT, mu=cfg.mu)
        print(f"N = {N}, seed {seed}: solved in {time.time() - t0:.0f} s", flush=True)
        for H in (cfg.H_grid[0], cfg.H_grid[-1]):
            lib = F._ns_make_weak_library(replace(cfg, H_xt=H, K=None), grid, variance_field=None, weak_seed=0)
            lib.fit([U])
            Theta, b = weak_system(lib, U)
            res, worst = [], 1.0
            for e in range(3):
                support = np.flatnonzero(C[e])
                x, *_ = np.linalg.lstsq(Theta[:, support], b[:, e], rcond=None)
                res.append(np.linalg.norm(Theta @ C[e] - b[:, e]) / np.linalg.norm(b[:, e]))
                worst = min(worst, float(np.min(np.abs(x))))
            print(f"  support {np.round(H, 3)}, K {lib.K}: residual at truth "
                  f"{np.round(100 * np.array(res), 2)} %, worst true coefficient {worst:.3f}, "
                  f"peak {peak_gb():.1f} GB", flush=True)


def memory(N: int, part: int, out: str) -> None:
    mod = __import__(f"main{part}")
    cfg = mod.make_config(N=N, n_runs=1, n_jobs=1, n_ensemble_models=5, results_dir=out,
                          results_filename=f"check_ns_part{part}.csv")
    t0 = time.time()
    mod.run_monte_carlo(cfg, verbose=False)
    print(f"N = {N}, Part {part}: peak {peak_gb():.1f} GB, {(time.time() - t0) / 60:.0f} min "
          "for one run with 5 bags", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "residual":
        residual(int(sys.argv[2]))
    else:
        memory(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
