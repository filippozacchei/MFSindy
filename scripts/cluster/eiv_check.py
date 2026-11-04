"""Validation of the plug-in errors-in-variables (EIV) score, every case, Parts I and II.

Not part of the reported results: it reads nothing from and writes nothing to
``examples/<case>/results``. It refits every run at every support exactly as the
paper's Monte Carlo does (the case's own builders, fits and seeds, captured from
its experiment function) and, for each rung, estimates from the noisy data and
the known noise model alone how much the noise in the library pulls each
excluded term into the fit:

    S = E[dA'dA], c = E[dA'dy]          first-order library noise (antithetic +-delta)
    M_A, m_y                            second-order offsets ((f(u+d)+f(u-d))/2 - f(u))
    g = c - S xi + A'(m_y - M_A xi)     expected EIV score at the fitted xi
    t_j = (g_j - G_jS G_SS^-1 g_S) / (s ||a_j projected off S||)

delta is drawn at the data's own noise level. |t_j| >> 1 says the EIV bias, not
the data, decides whether term j enters. ``t_sel`` uses the fit's own support
and coefficients (truth-free); ``t_true`` the true ones (a check of the
estimate, not available in practice).

    python scripts/cluster/eiv_check.py --cases lorenz --parts 1 --runs 2 --jobs 2
    python scripts/cluster/eiv_check.py --summary

One CSV per run in results/eiv_check/<case>_part<k>/, so a job can be resumed.
"""

from __future__ import annotations

import argparse
import importlib.util
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
OUT = ROOT / "results" / "eiv_check"
SMOKE = os.environ.get("EIV_SMOKE") == "1"
if SMOKE:
    OUT = OUT / "smoke"
CASES = ("lorenz", "hopf", "pendulum", "burgers", "vorticity", "isothermal_flow")

#: Case module, Part I and Part II experiment functions, and the Part II
#: function that draws a run's trajectories (it returns their variances last).
CASE_API = {
    "lorenz": ("lorenz", "run_lorenz_multi_trajectory_gls_experiment",
               "run_lorenz_intra_trajectory_gls_experiment", "lorenz_run_trajectories"),
    "hopf": ("hopf", "run_hopf_multi_trajectory_gls_experiment",
             "run_hopf_intra_trajectory_gls_experiment", "hopf_run_trajectories"),
    "pendulum": ("pendulum", "run_pendulum_multi_trajectory_gls_experiment",
                 "run_pendulum_intra_trajectory_gls_experiment", "pendulum_run_trajectories"),
    "burgers": ("burgers", "run_burgers_multi_trajectory_gls_experiment",
                "run_burgers_experiment", "burgers_run_trajectories"),
    "vorticity": ("vorticity_flow", "run_vorticity_multi_trajectory_gls_experiment",
                  "run_vorticity_intra_trajectory_gls_experiment", "vorticity_run_trajectories"),
    "isothermal_flow": ("isothermal_flow", "run_ns_isothermal_multi_trajectory_gls_experiment",
                        "run_ns_isothermal_intra_trajectory_gls_experiment", "ns_run_trajectories"),
}


def load_main(case: str, part: int):
    path = ROOT / "examples" / case / f"main{part}.py"
    spec = importlib.util.spec_from_file_location(f"eiv_main_{case}{part}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Captured(Exception):
    pass


def capture_driver(case: str, part: int, cfg):
    """The keyword arguments the case's experiment function hands the shared driver."""

    import importlib
    mod_name, run1, run2, _ = CASE_API[case]
    module = importlib.import_module(f"mfsindy.cases.{mod_name}")
    driver = "run_multi_trajectory_gls_experiment" if part == 1 else "run_intra_trajectory_gls_experiment"
    original = getattr(module, driver)
    store = {}

    def fake(cfg_, **kwargs):
        store.update(kwargs, cfg=cfg_)
        raise _Captured

    setattr(module, driver, fake)
    try:
        getattr(module, run1 if part == 1 else run2)(cfg)
    except _Captured:
        pass
    finally:
        setattr(module, driver, original)
    return module, store


# ----------------------------------------------------------------------------- systems

class _Dummy:
    coef_ = None

    def fit(self, theta, rhs, *args, **kwargs):
        return self


class _Recorder:
    """Optimizer factory recording each rung's stacked system, in fit order.

    With a factory the real optimizer fits as in the Monte Carlo; without, the
    systems are only recorded (the perturbed builds).
    """

    def __init__(self, factory=None):
        self.factory, self.systems = factory, []

    def __call__(self):
        opt = self.factory() if self.factory is not None else _Dummy()
        real_fit = opt.fit

        def fit(theta, rhs, *args, **kwargs):
            self.systems.append((np.asarray(theta, dtype=float), np.asarray(rhs, dtype=float)))
            return real_fit(theta, rhs, *args, **kwargs)

        opt.fit = fit
        return opt


def _per_equation(theta, r):
    return theta[r] if theta.ndim == 3 else theta


class Moments:
    """Running Monte Carlo moments of the library noise of one rung's system."""

    def __init__(self, theta0, rhs0):
        self.theta0, self.rhs0 = theta0, rhs0
        d, p = rhs0.shape[1], theta0.shape[-1]
        self.S = np.zeros((d, p, p))
        self.c = np.zeros((d, p))
        self.MA = np.zeros_like(theta0)
        self.my = np.zeros_like(rhs0)
        self.n = 0

    def add(self, plus, minus):
        (Ap, yp), (Am, ym) = plus, minus
        dA, dy = (Ap - Am) / 2, (yp - ym) / 2
        for r in range(self.rhs0.shape[1]):
            a = _per_equation(dA, r)
            self.S[r] += a.T @ a
            self.c[r] += a.T @ dy[:, r]
        self.MA += (Ap + Am) / 2 - self.theta0
        self.my += (yp + ym) / 2 - self.rhs0
        self.n += 1

    def scores(self, coef):
        """Standardised EIV score of every excluded term, per equation (zero on the support)."""

        n = max(self.n, 1)
        t = np.zeros_like(coef, dtype=float)
        for r in range(coef.shape[0]):
            A = _per_equation(self.theta0, r)
            MA = _per_equation(self.MA, r) / n
            y, my = self.rhs0[:, r], self.my[:, r] / n
            S, c = self.S[r] / n, self.c[r] / n
            sup = np.flatnonzero(coef[r] != 0)
            xi = coef[r, sup]
            g = c - S[:, sup] @ xi + A.T @ (my - MA[:, sup] @ xi)
            G = A.T @ A
            if sup.size:
                GSS = G[np.ix_(sup, sup)]
                gperp = g - G[:, sup] @ np.linalg.lstsq(GSS, g[sup], rcond=None)[0]
                norm2 = np.diag(G) - np.einsum("ij,ji->i", G[:, sup], np.linalg.lstsq(GSS, G[sup, :], rcond=None)[0])
                resid = y - A[:, sup] @ np.linalg.lstsq(A[:, sup], y, rcond=None)[0]
            else:
                gperp, norm2, resid = g, np.diag(G), y
            dof = max(len(y) - sup.size, 1)
            s = np.sqrt(max((resid ** 2).sum() / dof, 1e-300))
            # a term (almost) in the span of the support cannot be told apart from it
            free = norm2 > 1e-10 * np.maximum(np.diag(G), 1e-300)
            t[r, free] = gperp[free] / (s * np.sqrt(norm2[free]))
            t[r, sup] = 0.0
        return t


# ----------------------------------------------------------------------------- one run

def _orientation(post, coef):
    """Whether the case's post-processing transposes the optimizer's (equations, terms)."""

    probe = np.arange(coef.size, dtype=float).reshape(coef.shape)
    return np.array_equal(np.asarray(post(probe)), probe.T)


def _rows(case, part, run_idx, H, fits, moments, post, true, err_kwargs):
    from mfsindy.experiments.base import coefficient_errors
    rows = []
    for rung, opt in fits.items():
        coef = np.asarray(opt.coef_, dtype=float)
        transposed = _orientation(lambda c: post(c, rung), coef)
        true_eq = true.T if transposed else true
        L1, L0 = coefficient_errors(post(coef, rung), true, **err_kwargs(rung))
        m = moments[rung]
        t_sel = m.scores(coef)
        t_true = m.scores(np.where(np.abs(true_eq) > 1e-6, true_eq, 0.0))
        on, on_true = coef != 0, np.abs(true_eq) > 1e-6
        rows.append(dict(
            case=case, part=part, run=run_idx, support=str(H), rung=rung, selected=False,
            L1=L1, L0=L0, exact=L0 == 0, n_terms=int(on.sum()),
            spurious=int((on & ~on_true).sum()), missed=int((~on & on_true).sum()),
            tmax_sel=float(np.abs(t_sel).max()), n3_sel=int((np.abs(t_sel) > 3).sum()),
            tmax_true=float(np.abs(t_true).max()), n3_true=int((np.abs(t_true) > 3).sum()),
            draws=m.n,
        ))
    return rows


def _mark_selected(rows, trace):
    from mfsindy.experiments.selection import support_key
    chosen = {m: str(support_key(trace[m]["support"])) for m in trace}
    for row in rows:
        row["selected"] = chosen[row["rung"]] == row["support"]
    return rows


def run_part1(case, run_idx, draws, rng):
    from mfsindy.experiments.selection import select_rungs, support_key, support_variants
    from mfsindy.experiments.multi_trajectory import PART1_METHODS
    main = load_main(case, 1)
    cfg = main.make_config()
    if SMOKE:   # the largest support only, to exercise the code path
        cfg = replace(cfg, H_grid=[cfg.H_grid[-1]])
    _, kw = capture_driver(case, 1, cfg)
    cfg = kw["cfg"]
    methods = list(getattr(cfg, "methods", None) or PART1_METHODS)
    std = float(kw["reference_state_std"](cfg))
    sig = {"hf": cfg.noise_hf_rel * std, "lf": cfg.noise_lf_rel * std}
    # per-field noise scale (vorticity), read from the case's own dataset builder
    scale = inspect.getclosurevars(kw["dataset_builder"]).nonlocals.get("component_scale", 1.0)
    batch = kw["dataset_builder"](run_idx, cfg, sig["hf"], sig["lf"])
    post1 = kw.get("coef_postprocess") or (lambda c: c)
    post = lambda c, _r: post1(c)
    true = np.asarray(kw["true_coefficients"](batch, cfg))
    fit_fn = kw["fit_models_fn"]

    def systems_for(b, factory):
        rec = _Recorder(factory)
        out = fit_fn(b, variant, rec, t_argument=b.t_argument, noise_hf_abs=sig["hf"],
                     noise_lf_abs=sig["lf"], methods=methods)
        return out, dict(zip(methods, rec.systems))

    def perturbed(sign, deltas):
        return replace(batch, hf=[x + sign * d for x, d in zip(batch.hf, deltas["hf"])],
                       lf=[x + sign * d for x, d in zip(batch.lf, deltas["lf"])])

    fits, all_rows = {}, []
    for variant in support_variants(cfg):
        H = support_key(variant.H_xt)
        fits[H], systems = systems_for(batch, kw["optimizer_factory"])
        moments = {m: Moments(*systems[m]) for m in methods}
        state = np.random.get_state()
        for _ in range(draws):
            deltas = {g: [sig[g] * scale * rng.standard_normal(x.shape) for x in getattr(batch, g)]
                      for g in ("hf", "lf")}
            _, plus = systems_for(perturbed(+1, deltas), None)
            _, minus = systems_for(perturbed(-1, deltas), None)
            for m in methods:
                moments[m].add(plus[m], minus[m])
        np.random.set_state(state)
        all_rows += _rows(case, 1, run_idx, H, fits[H], moments, post, true, lambda _r: {})
        del moments, systems
    _, trace = select_rungs(fits, None)
    return _mark_selected(all_rows, trace)


def run_part2(case, run_idx, draws, rng):
    import importlib
    from mfsindy.experiments.intra_trajectory import fit_intra_trajectory_gls_optimizers
    from mfsindy.experiments.selection import select_rungs, support_key, support_variants
    from mfsindy.weighted_weak_pde_library import weak_system
    main = load_main(case, 2)
    cfg = main.make_config()
    if SMOKE:   # the largest support only, to exercise the code path
        cfg = replace(cfg, H_grid=[cfg.H_grid[-1]])
    module, kw = capture_driver(case, 2, cfg)
    cfg = kw["cfg"]
    methods = kw.get("methods") or getattr(cfg, "methods", None) or [
        "No weighting", "Ones GLS", "Variance diag", "Variance GLS"]
    post = kw.get("coef_postprocess") or (lambda c, _m: c)
    ek = kw.get("coefficient_error_kwargs")
    err_kwargs = (lambda m: dict(ek(m)) if callable(ek) else dict(ek or {}))

    # the variances of the drawn trajectories, from the case's own draw function
    draw_name = CASE_API[case][3]
    draw = getattr(module, draw_name)
    drawn = []

    def recording(*args, **kwargs):
        out = draw(*args, **kwargs)
        drawn.append((out[-2], out[-1]))
        return out

    setattr(module, draw_name, recording)
    try:
        np.random.seed((int(cfg.seed_base) + run_idx) % 2**32)   # as the Monte Carlo does
        fits, rows, true = {}, [], None
        for variant in support_variants(cfg):
            H = support_key(variant.H_xt)
            drawn.clear()
            artifacts = kw["run_builder"](run_idx, variant)
            artifacts = artifacts if isinstance(artifacts, list) else [artifacts]
            true = artifacts[0].true_coefficients
            fits[H] = fit_intra_trajectory_gls_optimizers(variant, artifacts, methods)
            noisy, variances = drawn[-1]
            sd = []
            for a, x, v in zip(artifacts, noisy, variances):
                if a.data.shape != np.shape(x) or not np.array_equal(a.data.ravel()[:50], np.ravel(x)[:50]):
                    raise RuntimeError(f"{case}: artifacts and drawn trajectories do not line up")
                v = np.asarray(v, dtype=float)
                sd.append(np.sqrt(v[..., None] if v.ndim == a.data.ndim - 1 else v))
            state = np.random.get_state()

            def systems(sign, deltas):
                out = {}
                for m in methods:
                    blocks = [weak_system(a.libraries[m], a.data + sign * d) for a, d in zip(artifacts, deltas)]
                    out[m] = (np.concatenate([b[0] for b in blocks], axis=-2),
                              np.concatenate([b[1] for b in blocks], axis=0))
                return out

            base = systems(0, [0.0] * len(artifacts))
            moments = {m: Moments(*base[m]) for m in methods}
            for _ in range(draws):
                deltas = [s * rng.standard_normal(a.data.shape) for a, s in zip(artifacts, sd)]
                plus, minus = systems(+1, deltas), systems(-1, deltas)
                for m in methods:
                    moments[m].add(plus[m], minus[m])
            np.random.set_state(state)
            rows += _rows(case, 2, run_idx, H, fits[H], moments, post, np.asarray(true), err_kwargs)
            del moments, base
    finally:
        setattr(module, draw_name, draw)
    _, trace = select_rungs(fits, post)
    return _mark_selected(rows, trace)


def one_run(case, part, run_idx, draws):
    path = OUT / f"{case}_part{part}" / f"run_{run_idx:03d}.csv"
    if path.exists():
        return str(path), "skipped", 0.0
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    import warnings
    warnings.filterwarnings("ignore")
    start = time.time()
    rng = np.random.default_rng(987_654 + 1000 * part + run_idx)
    rows = (run_part1 if part == 1 else run_part2)(case, run_idx, draws, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)
    return str(path), "done", time.time() - start


# ----------------------------------------------------------------------------- summary

def reproduction(case, part, df):
    """Largest gap between the refitted errors and the stored errors_by_support."""

    from mfsindy.experiments.selection import load_selection
    cfg = load_main(case, part).make_config()
    try:
        stored = load_selection(cfg)["errors_by_support"]
    except Exception:
        return np.nan
    gap = 0.0
    for row in df.itertuples():
        H = eval(row.support) if isinstance(row.support, str) else float(row.support)
        e = stored[row.rung][H][row.run]
        gap = max(gap, abs(e[0] - row.L1) / max(abs(e[0]), 1e-12), abs(e[1] - row.L0))
    return gap


def summary(threshold: float = 3.0):
    frames = [pd.read_csv(f) for f in sorted(OUT.glob("*_part*/run_*.csv"))]
    if not frames:
        print("no results in", OUT)
        return
    df = pd.concat(frames, ignore_index=True)
    df["flag"] = df.tmax_sel > threshold
    df["fail"] = df.spurious > 0
    print(f"flag = max|t_sel| > {threshold}; fail = at least one spurious term\n")
    for (case, part), g in df.groupby(["case", "part"], sort=False):
        print(f"== {case} Part {'I' if part == 1 else 'II'}: {g.run.nunique()} runs, "
              f"refit vs stored errors, max gap {reproduction(case, part, g):.2g}")
        pct = lambda x: f"{x.mean():.0%}" if len(x) else "-"
        lines = []
        for rung, r in g.groupby("rung", sort=False):
            sel = r[r.selected]
            lines.append(dict(
                rung=rung, fits=len(r), fail=int(r.fail.sum()),
                **{"flag|fail": pct(r[r.fail].flag), "flag|ok": pct(r[~r.fail].flag),
                   "sel fail": int(sel.fail.sum()), "sel flag|fail": pct(sel[sel.fail].flag),
                   "sel flag|ok": pct(sel[~sel.fail].flag),
                   "med t ok": round(r[~r.fail].tmax_sel.median(), 2),
                   "med t fail": round(r[r.fail].tmax_sel.median(), 2)}))
        table = pd.DataFrame(lines).set_index("rung")
        print(table.to_string(), "\n")
    df.to_csv(OUT / "all_runs.csv", index=False)
    print("all rows:", OUT / "all_runs.csv")


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--parts", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="largest support, 1 draw, results/eiv_check/smoke")
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()
    if args.smoke and not SMOKE:   # re-exec so workers and paths see it
        os.environ["EIV_SMOKE"] = "1"
        argv = [a for a in sys.argv if a != "--smoke"] + (["--draws", "1"] if "--draws" not in sys.argv else [])
        os.execv(sys.executable, [sys.executable] + argv)
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
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(args.jobs, mp_context=ctx) as pool:
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
