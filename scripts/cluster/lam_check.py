"""Validation of the a posteriori errors-in-variables diagnostics, every case, Parts I and II.

Not part of the reported results: it reads the stored selection records in
``examples/<case>/results`` and writes results/lam_check/ only. For every run and
rung it rebuilds the weak system at the support the Monte Carlo selected (the
case's own builders and seeds, as in eiv_check.py), adds M fresh draws of the
assumed noise to the data, and with dG, db the resulting changes of the system
evaluates, per equation and on the stored coefficients xi (support S):

    lambda_hat = lambda_max(S_SS, G_S'G_S),  S = mean dG'dG     reliability: noise vs signal on the fit's terms
    kappa_hat  = sum ||dG xi|| / sum ||db||                     feature- vs response-driven residual noise

No fit is repeated. ``ls_gap`` (least squares on the stored support against the
stored coefficients) checks that the rebuilt system is the one that was fitted:
its median over runs is about 1e-2 or less when data and noise level match (a
few bagged fits differ by up to ~0.25), order 1 if the results on disk come from
another noise setting. Run a case only after its Monte Carlo results are final.

    python scripts/cluster/lam_check.py --cases lorenz --parts 1 --runs 2 --jobs 2
    python scripts/cluster/lam_check.py --summary
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eiv_check import CASE_API, CASES, _Recorder, _orientation, _per_equation, capture_driver, load_main  # noqa: E402

OUT = ROOT / "results" / "lam_check"


def _lam_max(S, G):
    import scipy.linalg as sla
    if S.shape[0] == 0:
        return 0.0
    G = G + 1e-12 * max(np.trace(G) / len(G), 1e-300) * np.eye(len(G))
    return float(sla.eigh(S, G, eigvals_only=True)[-1])


def _as_raw(stored, d, p, post):
    """Stored coefficients in the optimizer's (equations, terms) layout.

    The case's post-processing decides, not the shape: with as many terms as
    equations (the pendulum) both layouts are square."""

    stored = np.asarray(stored, dtype=float)
    raw = stored.T if _orientation(post, np.zeros((d, p))) else stored
    if raw.shape != (d, p):
        raise ValueError(f"stored coefficients {stored.shape} do not match the system ({d}, {p})")
    return raw


class Accumulator:
    """dG'dG per equation and the numerator / denominator of kappa_hat, over draws."""

    def __init__(self, theta0, rhs0, coef):
        self.theta0, self.rhs0, self.coef = theta0, rhs0, coef
        d, p = rhs0.shape[1], theta0.shape[-1]
        self.S = np.zeros((d, p, p))
        self.num, self.den = np.zeros(d), np.zeros(d)
        self.n = 0

    def add(self, theta, rhs):
        dA, dy = theta - self.theta0, rhs - self.rhs0
        for r in range(self.rhs0.shape[1]):
            a = _per_equation(dA, r)
            self.S[r] += a.T @ a
            self.num[r] += np.linalg.norm(a @ self.coef[r])
            self.den[r] += np.linalg.norm(dy[:, r])
        self.n += 1

    def row(self, true_raw):
        lam, lam_true, gap = [], [], 0.0
        for r in range(self.rhs0.shape[1]):
            A = _per_equation(self.theta0, r)
            G, S = A.T @ A, self.S[r] / max(self.n, 1)
            sup = np.flatnonzero(self.coef[r])
            tru = np.flatnonzero(np.abs(true_raw[r]) > 1e-6)
            lam.append(_lam_max(S[np.ix_(sup, sup)], G[np.ix_(sup, sup)]))
            lam_true.append(_lam_max(S[np.ix_(tru, tru)], G[np.ix_(tru, tru)]))
            if sup.size:
                ls = np.linalg.lstsq(A[:, sup], self.rhs0[:, r], rcond=None)[0]
                gap = max(gap, np.linalg.norm(ls - self.coef[r, sup]) / max(np.linalg.norm(self.coef[r, sup]), 1e-300))
        kap = self.num / np.maximum(self.den, 1e-300)
        return dict(lam=max(lam), lam_eq=" ".join(f"{x:.3g}" for x in lam), lam_true=max(lam_true),
                    kappa=float(kap.max()), kappa_eq=" ".join(f"{x:.3g}" for x in kap), ls_gap=gap, draws=self.n)


def _row_head(case, part, run_idx, rung, H, coef_raw, true_raw):
    on, on_true = coef_raw != 0, np.abs(true_raw) > 1e-6
    spurious, missed = int((on & ~on_true).sum()), int((~on & on_true).sum())
    return dict(case=case, part=part, run=run_idx, rung=rung, support=str(H), n_terms=int(on.sum()),
                spurious=spurious, missed=missed, exact=spurious == 0 and missed == 0)


def _support_groups(rec, methods, run_idx):
    from mfsindy.experiments.selection import support_key
    groups = {}
    for m in methods:
        groups.setdefault(support_key(rec["supports"][m][run_idx]), []).append(m)
    return groups


def run_part1(case, run_idx, draws, rng):
    from mfsindy.experiments.multi_trajectory import PART1_METHODS
    from mfsindy.experiments.selection import load_selection, support_key, support_variants
    main = load_main(case, 1)
    cfg = main.make_config()
    rec = load_selection(cfg)
    _, kw = capture_driver(case, 1, cfg)
    cfg = kw["cfg"]
    methods = list(getattr(cfg, "methods", None) or PART1_METHODS)
    std = float(kw["reference_state_std"](cfg))
    sig = {"hf": cfg.noise_hf_rel * std, "lf": cfg.noise_lf_rel * std}
    scale = inspect.getclosurevars(kw["dataset_builder"]).nonlocals.get("component_scale", 1.0)
    batch = kw["dataset_builder"](run_idx, cfg, sig["hf"], sig["lf"])
    true = np.asarray(kw["true_coefficients"](batch, cfg))
    post = kw.get("coef_postprocess") or (lambda c: c)
    fit_fn = kw["fit_models_fn"]
    groups = _support_groups(rec, methods, run_idx)

    def systems_for(b, variant):
        recorder = _Recorder(None)
        fit_fn(b, variant, recorder, t_argument=b.t_argument, noise_hf_abs=sig["hf"],
               noise_lf_abs=sig["lf"], methods=methods)
        return dict(zip(methods, recorder.systems))

    rows = []
    for variant in support_variants(cfg):
        H = support_key(variant.H_xt)
        if H not in groups:
            continue
        base = systems_for(batch, variant)
        acc, heads = {}, {}
        for m in groups[H]:
            theta0, rhs0 = base[m]
            d, p = rhs0.shape[1], theta0.shape[-1]
            coef = _as_raw(rec["coefficients"][m][run_idx], d, p, post)
            true_raw = _as_raw(true, d, p, post)
            acc[m] = Accumulator(theta0, rhs0, coef)
            heads[m] = (_row_head(case, 1, run_idx, m, H, coef, true_raw), true_raw)
        del base
        for _ in range(draws):
            deltas = {g: [sig[g] * scale * rng.standard_normal(x.shape) for x in getattr(batch, g)]
                      for g in ("hf", "lf")}
            noisy = replace(batch, hf=[x + e for x, e in zip(batch.hf, deltas["hf"])],
                            lf=[x + e for x, e in zip(batch.lf, deltas["lf"])])
            pert = systems_for(noisy, variant)
            for m in groups[H]:
                acc[m].add(*pert[m])
            del pert
        for m in groups[H]:
            head, true_raw = heads[m]
            rows.append({**head, **acc[m].row(true_raw)})
    return rows


def run_part2(case, run_idx, draws, rng):
    from mfsindy.experiments.selection import load_selection, support_key, support_variants
    from mfsindy.weighted_weak_pde_library import weak_system
    main = load_main(case, 2)
    cfg = main.make_config()
    rec = load_selection(cfg)
    module, kw = capture_driver(case, 2, cfg)
    cfg = kw["cfg"]
    methods = kw.get("methods") or getattr(cfg, "methods", None) or [
        "No weighting", "Ones GLS", "Variance diag", "Variance GLS"]
    groups = _support_groups(rec, methods, run_idx)
    post2 = kw.get("coef_postprocess") or (lambda c, _m: c)

    draw_name = CASE_API[case][3]
    draw = getattr(module, draw_name)
    drawn = []

    def recording(*args, **kwargs):
        out = draw(*args, **kwargs)
        drawn.append((out[-2], out[-1]))
        return out

    setattr(module, draw_name, recording)
    rows = []
    try:
        for variant in support_variants(cfg):
            H = support_key(variant.H_xt)
            if H not in groups:
                continue
            np.random.seed((int(cfg.seed_base) + run_idx) % 2**32)   # as the Monte Carlo does
            drawn.clear()
            artifacts = kw["run_builder"](run_idx, variant)
            artifacts = artifacts if isinstance(artifacts, list) else [artifacts]
            true = np.asarray(artifacts[0].true_coefficients)
            noisy, variances = drawn[-1]
            sd = []
            for a, x, v in zip(artifacts, noisy, variances):
                if a.data.shape != np.shape(x) or not np.array_equal(a.data.ravel()[:50], np.ravel(x)[:50]):
                    raise RuntimeError(f"{case}: artifacts and drawn trajectories do not line up")
                v = np.asarray(v, dtype=float)
                sd.append(np.sqrt(v[..., None] if v.ndim == a.data.ndim - 1 else v))

            def system(m, deltas):
                blocks = [weak_system(a.libraries[m], a.data + dd) for a, dd in zip(artifacts, deltas)]
                return (np.concatenate([b[0] for b in blocks], axis=-2),
                        np.concatenate([b[1] for b in blocks], axis=0))

            acc, heads = {}, {}
            for m in groups[H]:
                for a in artifacts:   # as fit_intra_trajectory_gls_optimizers does before assembling
                    a.libraries[m].fit([a.data])
                theta0, rhs0 = system(m, [0.0] * len(artifacts))
                d, p = rhs0.shape[1], theta0.shape[-1]
                post = lambda c, m=m: post2(c, m)
                coef = _as_raw(rec["coefficients"][m][run_idx], d, p, post)
                true_raw = _as_raw(true, d, p, post)
                acc[m] = Accumulator(theta0, rhs0, coef)
                heads[m] = (_row_head(case, 2, run_idx, m, H, coef, true_raw), true_raw)
            for _ in range(draws):
                deltas = [s * rng.standard_normal(a.data.shape) for a, s in zip(artifacts, sd)]
                for m in groups[H]:
                    acc[m].add(*system(m, deltas))
            for m in groups[H]:
                head, true_raw = heads[m]
                rows.append({**head, **acc[m].row(true_raw)})
            del artifacts
    finally:
        setattr(module, draw_name, draw)
    return rows


def one_run(case, part, run_idx, draws):
    path = OUT / f"{case}_part{part}" / f"run_{run_idx:03d}.csv"
    if path.exists():
        return str(path), "skipped", 0.0
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    import warnings
    warnings.filterwarnings("ignore")
    start = time.time()
    rng = np.random.default_rng(424_242 + 1000 * part + run_idx)
    rows = (run_part1 if part == 1 else run_part2)(case, run_idx, draws, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)
    return str(path), "done", time.time() - start


def summary(threshold: float):
    frames = [pd.read_csv(f) for f in sorted(OUT.glob("*_part*/run_*.csv"))]
    if not frames:
        print("no results in", OUT)
        return
    df = pd.concat(frames, ignore_index=True)
    df["flag"] = df.lam > threshold
    df["fail"] = df.spurious > 0
    print(f"flag = lambda_hat on the selected support > {threshold}; fail = at least one spurious term\n")
    pct = lambda x: f"{x.mean():.0%} ({int(x.sum())}/{len(x)})" if len(x) else "-"
    for (case, part), g in df.groupby(["case", "part"], sort=False):
        print(f"== {case} Part {'I' if part == 1 else 'II'}: {g.run.nunique()} runs, "
              f"median ls_gap {g.ls_gap.median():.2g} (about 1e-2 or less: the stored fits were rebuilt; "
              f"order 1: the results on disk come from another noise setting)")
        lines = []
        for rung, r in g.groupby("rung", sort=False):
            lines.append({"rung": rung, "fail": f"{int(r.fail.sum())}/{len(r)}",
                          "flag|fail": pct(r[r.fail].flag), "flag|ok": pct(r[~r.fail].flag),
                          "med lam ok": round(r[~r.fail].lam.median(), 3) if (~r.fail).any() else np.nan,
                          "med lam fail": round(r[r.fail].lam.median(), 3) if r.fail.any() else np.nan,
                          "med lam true": round(r.lam_true.median(), 3),
                          "med kappa": round(r.kappa.median(), 3)})
        print(pd.DataFrame(lines).set_index("rung").to_string(), "\n")
    print(f"ALL: flag|fail {pct(df[df.fail].flag)}, flag|ok {pct(df[~df.fail].flag)}")
    df.to_csv(OUT / "all_runs.csv", index=False)
    print("all rows:", OUT / "all_runs.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--parts", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    # Fixed before the validation: at 0.5 the bias bound lambda / (1 - lambda) equals the coefficient.
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    if args.summary:
        summary(args.threshold)
        return
    tasks = [(c, p, r) for c in args.cases for p in args.parts for r in range(args.first, args.first + args.runs)]
    failed = 0
    if args.jobs == 1:
        for c, p, r in tasks:
            try:
                path, status, sec = one_run(c, p, r, args.draws)
                print(f"{c} part {p} run {r}: {status} ({sec:.0f}s)", flush=True)
            except Exception:
                failed += 1
                print(f"!! {c} part {p} run {r} failed\n{traceback.format_exc()}", flush=True)
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("fork")) as pool:
            futures = {pool.submit(one_run, c, p, r, args.draws): (c, p, r) for c, p, r in tasks}
            for future in as_completed(futures):
                c, p, r = futures[future]
                try:
                    path, status, sec = future.result()
                    print(f"{c} part {p} run {r}: {status} ({sec:.0f}s)", flush=True)
                except Exception:
                    failed += 1
                    print(f"!! {c} part {p} run {r} failed\n{traceback.format_exc()}", flush=True)
    print(f"finished, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
