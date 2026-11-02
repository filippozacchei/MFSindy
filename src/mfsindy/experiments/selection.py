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
  bagged least-squares coefficients, before any thresholding, are tightest: the
  median, over every term of the library, of their interquartile range relative
  to the largest coefficient of the equation. The support is judged on the
  unthresholded fit so the threshold cannot influence it: measured on the kept
  terms only, a support where BIC cuts true terms (on Hopf, the near-cancelling
  radial terms) left a sparse model that the bags agree on closely, and won.
  A support is admissible only if its weak system has at least twice as many
  rows as library terms: below that, the minimum-norm bags of an underdetermined
  system agree with each other without identifying anything, and the spread
  cannot see it (single-trajectory HF at the widest supports).

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


def _ridge(G: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """argmin |b - G w|^2 + alpha |w|^2, as least squares on the design augmented
    by sqrt(alpha) I, which keeps the conditioning of G rather than squaring it as
    the normal equations would."""

    p = G.shape[1]
    G_aug = np.vstack([G, np.sqrt(alpha) * np.eye(p)])
    b_aug = np.concatenate([b, np.zeros(p)])
    return np.linalg.lstsq(G_aug, b_aug, rcond=None)[0]


def _stlsq(G: np.ndarray, b: np.ndarray, w: np.ndarray, threshold: float, max_iter: int,
           alpha: float) -> np.ndarray:
    """Sequential thresholded ridge regression (STRidge) for one equation, started
    from ``w``, the ridge fit of ``b`` on all of ``G``."""

    keep = np.ones(G.shape[1], dtype=bool)
    for it in range(max_iter):
        new = np.abs(w) >= threshold
        if not new.any():
            return np.zeros_like(w)
        if it > 0 and np.array_equal(new, keep):
            break
        keep = new
        w = np.zeros_like(w)
        w[keep] = _ridge(G[:, keep], b, alpha)
    return w


#: A support is admissible when its weak system has at least this many rows per
#: library term.
MIN_ROWS_PER_TERM = 2


class BICEnsembleSTLSQ(BaseOptimizer):
    """Bagged STRidge whose threshold is chosen per equation by BIC.

    The ridge ``alpha`` is the paper's lambda_2, small enough (1e-12) only to keep
    the refits well posed; pysindy's default of 0.05 biases the weak fits, and
    falls unevenly on whitened and unwhitened systems (see the Lorenz config).

    The thresholds span 1e-3 to 1 times the largest least-squares coefficient of
    each equation, log-spaced. Rows are bagged with replacement from the global
    RNG, which the multi-trajectory pipeline seeds per rung. After ``fit``:
    ``coef_list`` holds the bags at each equation's chosen threshold, ``coef_``
    their median, ``selection_`` the threshold path (thresholds, RSS, number of
    terms and BIC per threshold and equation, and the chosen index), and
    ``spread_`` the support-selection criterion.

    ``nested_sizes`` lets BIC choose the library as well, among nested ones made
    of the first ``s`` columns (a polynomial library ordered by degree, for each
    degree). Every size runs its own threshold path on the same rows, so the BIC
    values are comparable, and each equation keeps the (size, threshold) pair
    that minimises it. A size is a candidate only where it meets the row cap
    (``MIN_ROWS_PER_TERM``). The spread, which chooses the support, is measured
    on one reference library of ``spread_size`` columns at every support, and
    the row cap that rules supports out is that library's: measured on the
    chosen size instead, it favoured the long supports where the cap leaves only
    the smallest library, whose few coefficients the bags agree on.
    """

    def __init__(self, n_models: int = 100, n_thresholds: int = 20, max_iter: int = 20,
                 alpha: float = 1e-12, nested_sizes=None, spread_size=None):
        super().__init__(max_iter=max_iter, unbias=False)
        self.n_models = n_models
        self.n_thresholds = n_thresholds
        self.alpha = alpha
        self.nested_sizes = nested_sizes
        self.spread_size = spread_size

    def fit(self, x, y, **kwargs):
        """Fit on one shared design, or on one design per equation.

        ``x`` of shape (d, n, p) gives equation j its own design ``x[j]``, as a
        library that whitens each equation separately produces; the bags still
        draw the same rows (test functions) for every equation.
        """

        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            return super().fit(x, y, **kwargs)
        y = np.asarray(y, dtype=float)
        y = y[:, None] if y.ndim == 1 else y
        if x.shape[0] != y.shape[1] or x.shape[1] != y.shape[0]:
            raise ValueError(f"per-equation designs {x.shape} do not match targets {y.shape}.")
        self._solve(list(x), y)
        self.ind_ = np.abs(self.coef_) > 1e-14
        return self

    def _reduce(self, x, y):
        self._solve([x] * y.shape[1], y)

    def _path(self, designs, y, bags):
        """The threshold path on ``designs``: bags at every threshold, their
        unthresholded fits, and the median model's RSS, size and BIC."""

        n, p = designs[0].shape
        t = y.shape[1]
        scale = np.array([np.max(np.abs(_ridge(designs[j], y[:, j], self.alpha))) for j in range(t)])
        thresholds = np.geomspace(1e-3, 1.0, self.n_thresholds)[:, None] * scale[None, :]

        coefs = np.zeros((self.n_thresholds, self.n_models, t, p))
        ls = np.zeros((self.n_models, t, p))
        # A bag's rows and its ridge fit are the same at every threshold,
        # so both are made once per bag (the fit was most of a run's time).
        for j in range(t):
            for k, rows in enumerate(bags):
                G, b = designs[j][rows], y[rows, j]
                ls[k, j] = _ridge(G, b, self.alpha)
                for i in range(self.n_thresholds):
                    coefs[i, k, j] = _stlsq(G, b, ls[k, j], thresholds[i, j], self.max_iter, self.alpha)

        median = np.median(coefs, axis=1)                                  # (n_thr, t, p)
        rss = np.stack([np.sum((designs[j] @ median[:, j].T - y[:, [j]]) ** 2, axis=0)
                        for j in range(t)], axis=1)                        # (n_thr, t)
        nnz = np.count_nonzero(median, axis=2)
        bic = n * np.log(np.maximum(rss, 1e-300) / n) + nnz * np.log(n)
        return thresholds, coefs, ls, median, rss, nnz, bic

    def _solve(self, designs, y):
        n, p = designs[0].shape
        t = y.shape[1]
        bags = [np.random.randint(0, n, n) for _ in range(self.n_models)]
        if self.nested_sizes is None:
            thresholds, coefs, ls, median, rss, nnz, bic = self._path(designs, y, bags)
            choice = np.argmin(bic, axis=0)                                # first minimiser
            self.n_rows_, self.n_terms_ = n, p
            self.ls_coef_list = ls
            self._spread_terms = np.ones((t, p), dtype=bool)
            self.coef_list = np.stack([coefs[choice[j], :, j] for j in range(t)], axis=1)
            self.coef_ = np.stack([median[choice[j], j] for j in range(t)])
            self.selection_ = {"thresholds": thresholds, "rss": rss, "nnz": nnz, "bic": bic,
                               "choice": choice}
            return

        sizes = sorted(int(s) for s in self.nested_sizes)
        # Below the cap no size is a candidate; the smallest is fitted so the
        # support still has a record, and select_support then rules it out.
        sizes = [s for s in sizes if n >= MIN_ROWS_PER_TERM * s] or sizes[:1]
        paths = []
        for s in sizes:
            thresholds, coefs, ls, median, rss, nnz, bic = self._path([d[:, :s] for d in designs], y, bags)
            pad = ((0, 0),) * (coefs.ndim - 1) + ((0, p - s),)
            paths.append((thresholds, np.pad(coefs, pad), np.pad(ls, pad[1:]),
                          np.pad(median, pad[1:]), rss, nnz, bic))
        thresholds, coefs, ls, median, rss, nnz, bic = (np.concatenate(parts) for parts in zip(*paths))
        size_of = np.repeat(sizes, self.n_thresholds)                      # per candidate
        choice = np.argmin(bic, axis=0)                                    # first minimiser
        chosen = size_of[choice]                                           # (t,)

        # The unthresholded bags on the reference library, for the spread.
        reference = int(self.spread_size or max(self.nested_sizes))
        self.n_rows_, self.n_terms_ = n, reference
        if reference in sizes:
            self.ls_coef_list = paths[sizes.index(reference)][2]
        else:
            self.ls_coef_list = np.zeros((self.n_models, t, p))
            for j in range(t):
                for k, rows in enumerate(bags):
                    self.ls_coef_list[k, j, :reference] = _ridge(designs[j][rows, :reference],
                                                                 y[rows, j], self.alpha)
        self._spread_terms = np.broadcast_to(np.arange(p) < reference, (t, p))
        self.coef_list = np.stack([coefs[choice[j], :, j] for j in range(t)], axis=1)
        self.coef_ = np.stack([median[choice[j], j] for j in range(t)])
        self.selection_ = {"thresholds": thresholds, "rss": rss, "nnz": nnz, "bic": bic,
                           "choice": choice, "sizes": size_of, "size": chosen}

    @property
    def spread_(self) -> float:
        """Median IQR of the bagged least-squares coefficients, before thresholding,
        each relative to the largest coefficient magnitude of its equation, over
        the terms of the reference library."""

        q75, q50, q25 = np.percentile(self.ls_coef_list, [75, 50, 25], axis=0)   # (t, p)
        scale = np.max(np.abs(q50), axis=1, keepdims=True)
        relative = (q75 - q25) / np.maximum(scale, 1e-300)
        return float(np.median(relative[self._spread_terms]))


def select_support(fits_by_support: Mapping[Any, Mapping[str, Any]]) -> Dict[str, Any]:
    """Per rung, the admissible support whose unthresholded bags have the smallest spread.

    ``fits_by_support`` maps each candidate support, as :func:`support_key`, to
    ``{rung: optimizer}``. Admissible supports have at least
    ``MIN_ROWS_PER_TERM`` rows per library term; if none is, every support is a
    candidate. Ties keep the first support in grid order.
    """

    supports = list(fits_by_support)
    rungs = list(fits_by_support[supports[0]])
    chosen = {}
    for rung in rungs:
        fits = {H: fits_by_support[H][rung] for H in supports}
        admissible = [H for H, f in fits.items() if f.n_rows_ >= MIN_ROWS_PER_TERM * f.n_terms_]
        chosen[rung] = min(admissible or supports, key=lambda H: fits[H].spread_)
    return chosen


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
                   coefficients: Mapping[str, list] | None = None,
                   supports: Mapping[str, list] | None = None,
                   errors_by_support: Mapping[str, Mapping[Any, np.ndarray]] | None = None) -> None:
    """Write the last run's selection, and every run's reported coefficients and
    chosen support; with ``errors_by_support``, every run's (L1, L0) at every
    support, per rung, as {rung: {support: array (n_runs, 2)}}."""

    path = selection_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"run": run_idx, "true": np.asarray(true_coefficients), "rungs": trace}
    if coefficients is not None:
        record["coefficients"] = {rung: np.stack(c) for rung, c in coefficients.items()}
    if supports is not None:
        record["supports"] = {rung: list(s) for rung, s in supports.items()}
    if errors_by_support is not None:
        record["errors_by_support"] = {rung: dict(e) for rung, e in errors_by_support.items()}
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
