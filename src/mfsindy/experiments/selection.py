"""Hyperparameter selection inside the ensemble: threshold by BIC, support by spread.

Both choices are made by the estimator itself, from the system it is given, with
no validation data and no true coefficients, so every rung goes through the same
procedure on its own (whitened, where weighted) weak system.

- Threshold. Every bag is fitted by STLSQ at every threshold of a grid; the
  median model at each threshold is scored on the full system by
  BIC = N log(RSS/N) + k log N (Mangan et al. 2017), and each equation keeps the
  threshold that minimises it. BIC picks a point on the fit-complexity Pareto
  front, with the trade-off fixed by log N.
- Support. The rung is fitted at every support of a grid and keeps the one whose
  bagged coefficients are tightest: the median, over the terms kept, of their
  interquartile range relative to their magnitude.

Tested against the alternatives on the ODE benchmarks (scratch/esindy, scratch/
support, scratch/mstls): WSINDy's MSTLS drops true terms when the library is small
or terms nearly cancel; Tran-Bortz radius selection controls quadrature error
only and ignores the noise; held-out and leave-one-out scores either saturate or
overfit. BIC stays within 1.3x of the oracle threshold for every rung, and the
spread rule within 1.3x of each rung's best support.
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
from pysindy.optimizers import BaseOptimizer


def _stlsq(G: np.ndarray, b: np.ndarray, threshold: float, max_iter: int) -> np.ndarray:
    """Sequential thresholding least squares for one equation, no ridge."""

    w = np.linalg.lstsq(G, b, rcond=None)[0]
    keep = np.ones(G.shape[1], dtype=bool)
    for it in range(max_iter):
        new = np.abs(w) >= threshold
        if not new.any():
            return np.zeros_like(w)
        if it > 0 and np.array_equal(new, keep):
            break
        keep = new
        w = np.zeros_like(w)
        w[keep] = np.linalg.lstsq(G[:, keep], b, rcond=None)[0]
    return w


class BICEnsembleSTLSQ(BaseOptimizer):
    """Bagged STLSQ whose threshold is chosen per equation by BIC.

    The thresholds span 1e-3 to 1 times the largest least-squares coefficient of
    each equation, log-spaced. Rows are bagged with replacement from the global
    RNG, which the multi-trajectory pipeline seeds per rung. After ``fit``:
    ``coef_list`` holds the bags at each equation's chosen threshold, ``coef_``
    their median, ``selection_`` the threshold path (thresholds, RSS, number of
    terms and BIC per threshold and equation, and the chosen index), and
    ``spread_`` the support-selection criterion.
    """

    def __init__(self, n_models: int = 100, n_thresholds: int = 20, max_iter: int = 20):
        super().__init__(max_iter=max_iter, unbias=False)
        self.n_models = n_models
        self.n_thresholds = n_thresholds

    def _reduce(self, x, y):
        n, p = x.shape
        t = y.shape[1]
        bags = [np.random.randint(0, n, n) for _ in range(self.n_models)]
        w_ls = np.linalg.lstsq(x, y, rcond=None)[0]                       # (p, t)
        scale = np.max(np.abs(w_ls), axis=0)
        thresholds = np.geomspace(1e-3, 1.0, self.n_thresholds)[:, None] * scale[None, :]

        coefs = np.zeros((self.n_thresholds, self.n_models, t, p))
        for j in range(t):
            for i in range(self.n_thresholds):
                for k, rows in enumerate(bags):
                    coefs[i, k, j] = _stlsq(x[rows], y[rows, j], thresholds[i, j], self.max_iter)

        median = np.median(coefs, axis=1)                                  # (n_thr, t, p)
        rss = np.sum((np.einsum("np,itp->itn", x, median) - y.T[None]) ** 2, axis=2)
        nnz = np.count_nonzero(median, axis=2)
        bic = n * np.log(np.maximum(rss, 1e-300) / n) + nnz * np.log(n)
        choice = np.argmin(bic, axis=0)                                    # first minimiser

        self.coef_list = np.stack([coefs[choice[j], :, j] for j in range(t)], axis=1)
        self.coef_ = np.stack([median[choice[j], j] for j in range(t)])
        self.selection_ = {"thresholds": thresholds, "rss": rss, "nnz": nnz, "bic": bic,
                           "choice": choice}

    @property
    def spread_(self) -> float:
        """Median relative IQR of the bagged coefficients over the terms kept."""

        kept = self.coef_ != 0
        if not kept.any():
            return float("inf")
        q75, q25 = np.percentile(self.coef_list, [75, 25], axis=0)
        return float(np.median((q75 - q25)[kept] / np.abs(self.coef_[kept])))


def select_support(fits_by_support: Mapping[Any, Mapping[str, Any]]) -> Dict[str, Any]:
    """Per rung, the support whose fitted optimizer has the smallest spread.

    ``fits_by_support`` maps each candidate support, as :func:`support_key`, to
    ``{rung: optimizer}``. Ties keep the first support in grid order.
    """

    supports = list(fits_by_support)
    rungs = list(fits_by_support[supports[0]])
    return {rung: min(supports, key=lambda H: fits_by_support[H][rung].spread_) for rung in rungs}


def selection_record(optimizer: BICEnsembleSTLSQ) -> Dict[str, Any]:
    """What a fitted optimizer chose, for plotting: the path and the spread."""

    return {"spread": optimizer.spread_, **{k: np.asarray(v) for k, v in optimizer.selection_.items()}}


def support_key(H: Any) -> Any:
    """A hashable form of a support (a scalar, or a list per dimension for PDEs)."""

    return tuple(np.atleast_1d(H).tolist()) if np.ndim(H) else float(H)


def support_variants(cfg) -> list:
    """The configs a run is fitted under: one per support of ``H_grid``.

    K is derived from each support, at the same coverage, so the grid varies the
    support alone. Without a grid the config is used as given.
    """

    if getattr(cfg, "H_grid", None) is None:
        return [cfg]
    return [replace(cfg, H_xt=H, K=None) for H in cfg.H_grid]


def select_rungs(
    fits_by_support: Mapping[Any, Mapping[str, BICEnsembleSTLSQ]],
    postprocess: Callable[[np.ndarray, str], np.ndarray] | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Each rung's coefficients at its chosen support, and a record of the choice.

    The record keeps, per rung, the chosen support, the threshold path and spread
    at every support, and the bagged coefficients at the chosen one -- what the
    selection and coefficient-distribution plots draw.
    """

    post = postprocess or (lambda coef, _rung: coef)
    coefs, trace = {}, {}
    for rung, key in select_support(fits_by_support).items():
        optimizer = fits_by_support[key][rung]
        coefs[rung] = post(np.asarray(optimizer.coef_), rung)
        trace[rung] = {
            "support": key,
            "by_support": {k: selection_record(f[rung]) for k, f in fits_by_support.items()},
            "bags": np.stack([post(np.asarray(c), rung) for c in optimizer.coef_list]),
        }
    return coefs, trace


def selection_path(cfg) -> Path:
    """Where a Monte Carlo writes the record of its last run's selection."""

    return Path(cfg.results_dir) / f"{Path(cfg.results_filename).stem}_selection.pkl"


def save_selection(cfg, run_idx: int, true_coefficients: np.ndarray, trace,
                   coefficients: Mapping[str, list] | None = None) -> None:
    """Write the last run's selection, and every run's reported coefficients."""

    path = selection_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"run": run_idx, "true": np.asarray(true_coefficients), "rungs": trace}
    if coefficients is not None:
        record["coefficients"] = {rung: np.stack(c) for rung, c in coefficients.items()}
    with open(path, "wb") as fh:
        pickle.dump(record, fh)


def load_selection(cfg) -> Dict[str, Any]:
    with open(selection_path(cfg), "rb") as fh:
        return pickle.load(fh)


__all__ = [
    "BICEnsembleSTLSQ",
    "load_selection",
    "save_selection",
    "select_rungs",
    "select_support",
    "selection_path",
    "selection_record",
    "support_key",
    "support_variants",
]
