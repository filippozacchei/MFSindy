import warnings

from typing import Sequence

import numpy as np
from pysindy.feature_library.weak_pde_library import WeakPDELibrary
from pysindy.utils import AxesArray


class WeakCovarianceWarning(UserWarning):
    """The weak covariance is singular, so whitening by it is not meaningful."""


def drop_duplicate_domains(library) -> int:
    """Remove test functions whose support is an exact copy of another's.

    pysindy draws the K domain centres uniformly at random and then recentres
    each onto grid points, so two centres falling in the same grid interval end
    up selecting identical samples with identical weights -- byte-identical rows
    of the weak system. By the birthday argument these are common well before K
    approaches the number of grid positions: at 100 Lorenz samples, 99 requested
    test functions collapse to 54 distinct ones.

    A duplicated row adds no information in any formulation, but it is not
    harmless here: it makes the weak covariance exactly singular, and whitening
    then divides that direction by the nugget, amplifying round-off by ~1e6.
    Dropping the copies is bookkeeping rather than a modelling choice, so it
    happens for every weak library, weighted or not, keeping the rungs on a
    common set of test functions.
    """

    n_requested = int(library.K)
    seen: set = set()
    keep: list[int] = []
    for k in range(n_requested):
        support = tuple(
            tuple(np.asarray(axis).ravel().tolist()) for axis in library.inds_k[k]
        )
        if support in seen:
            continue
        seen.add(support)
        keep.append(k)

    dropped = n_requested - len(keep)
    if dropped:
        for name in ("inds_k", "fulltweights", "fullweights0", "fullweights1"):
            sequence = getattr(library, name, None)
            if sequence is not None and len(sequence) == n_requested:
                setattr(library, name, [sequence[i] for i in keep])
        library.K = len(keep)
    return dropped


class DedupedWeakPDELibrary(WeakPDELibrary):
    """WeakPDELibrary that discards duplicate test-function supports.

    ``deduplicate=False`` restores pysindy's behaviour, keeping every requested
    test function including exact copies. That is not a useful way to fit, since
    the copies make the weak covariance exactly singular, but it is the only way
    to measure what the duplicates cost: the comparison needs a run with them.
    """

    def __init__(self, *args, deduplicate: bool = True, **kwargs):
        self.deduplicate = deduplicate
        super().__init__(*args, **kwargs)

    def _weak_form_setup(self):
        super()._weak_form_setup()
        self.n_duplicate_domains_ = (
            drop_duplicate_domains(self) if self.deduplicate else 0
        )
        
def weak_validity_ratio(library, jacobian_norm) -> np.ndarray:
    """kappa per test function: how much of the residual the covariance ignores.

    The covariance model keeps only the term in which noise enters through the
    test function's derivative, and neglects the one carrying the library
    Jacobian. The approximation holds where the neglected term is small against
    the retained one, which the appendix writes as a ratio of squared norms,
    ``kappa = || phi |grad F| ||^2 / || phi_dot ||^2``, predicted to scale as
    O(h^2) in the support width. Returned per test function so the spread across
    the weak system is visible, not just its centre.

    ``jacobian_norm`` is |grad F(u)| sampled along the trajectory, one value per
    grid point, so it is the caller's business: each system knows its own
    Jacobian.
    """

    jacobian_norm = np.asarray(jacobian_norm, dtype=float).ravel()
    ratios = []
    for k in range(int(library.K)):
        indices = np.asarray(library.inds_k[k][0]).ravel()
        phi_dot = np.asarray(library.fulltweights[k], dtype=float).ravel()
        phi = np.asarray(library.fullweights0[k], dtype=float).ravel()
        denominator = float(np.sum(phi_dot**2))
        if denominator <= 0.0:
            continue
        ratios.append(float(np.sum((phi * jacobian_norm[indices]) ** 2) / denominator))
    return np.asarray(ratios)



def pde_feature_weight_index(library) -> np.ndarray:
    """Which test-function weight array carries each output feature.

    In the weak form a library term's spatial derivatives are integrated by
    parts onto the test function, so the noise in that term reaches the residual
    through some ``d^alpha phi`` rather than through ``phi`` itself. Kappa needs
    to know which one, per feature. The answer is fixed by how pysindy lays the
    output features out:

    * an optional bias and the plain library functions keep ``phi`` itself;
    * the pure integral terms ``u_n`` with multi-index ``alpha_j`` take
      ``d^alpha_j phi``, i.e. ``fullweights1[j]``;
    * the mixed terms transfer only half of the derivative onto the test
      function -- ``derivs_mixed = multiindices[j] // 2`` in pysindy's product
      rule -- and the rest stays on the data.

    Returns one index per output feature: ``-1`` for ``fullweights0`` (plain
    ``phi``), otherwise the index into ``fullweights1``.
    """

    multiindices = np.asarray(library.multiindices)
    num_derivatives = int(library.num_derivatives)
    n_features = int(library.n_features_in_)
    include_bias = bool(library.include_bias)
    include_interaction = bool(getattr(library, "include_interaction", True))

    # The function library's term count is not stored, but the output width
    # pins it down given the blocks below.
    n_output = int(library.n_output_features_)
    divisor = 1 + (num_derivatives * n_features if include_interaction else 0)
    n_library_terms = (
        n_output - int(include_bias) - num_derivatives * n_features
    ) // divisor

    def index_of(multiindex) -> int:
        if not np.any(multiindex):
            return -1
        matches = np.where(np.all(multiindices == multiindex, axis=1))[0]
        if matches.size == 0:
            # No weight array holds this order; fall back to plain phi, which
            # understates kappa rather than inventing a weight.
            return -1
        return int(matches[0])

    order: list[int] = []
    if include_bias:
        order.append(-1)
    order.extend([-1] * n_library_terms)
    for j in range(num_derivatives):
        order.extend([index_of(multiindices[j])] * n_features)
    if include_interaction:
        for j in range(num_derivatives):
            mixed = multiindices[j] // 2
            order.extend([index_of(mixed)] * (n_features * n_library_terms))

    if len(order) != n_output:
        raise ValueError(
            f"Feature layout does not add up: derived {len(order)} entries for "
            f"{n_output} output features."
        )
    return np.asarray(order, dtype=int)


def pde_weak_validity_ratio(library, sensitivity_fields) -> np.ndarray:
    """kappa per test function for a PDE weak system.

    The PDE counterpart of :func:`weak_validity_ratio`, and computed the same
    way -- from the assembled system, as a ratio of squared norms -- rather than
    from the support-width scaling. The scaling form ``h_t^2 h_x^(-2m)`` carries
    units, so it cannot be compared between benchmarks or against the ODE kappa
    and has no absolute threshold to test; this one is dimensionless and does.

    The neglected term reaches residual row ``k`` through ``d^alpha phi_k``
    weighted by each feature's sensitivity to the state, so the effective weight
    on the noise is ``sum_p |d^alpha_p phi_k| * s_p``. Summing magnitudes is a
    triangle-inequality bound: cancellation between features would only make the
    true ratio smaller, so kappa is an upper bound and never flatters the
    covariance model.

    ``sensitivity_fields`` has shape ``(n_output_features, *grid_shape)`` and
    carries ``|Xi_p| * |d(feature_p)/du|`` on the grid -- the coefficient
    included, since a feature the model does not use cannot break it. It is the
    caller's business, as the library map is: each system knows its own.
    """

    sensitivity_fields = np.asarray(sensitivity_fields, dtype=float)
    weight_index = pde_feature_weight_index(library)
    if sensitivity_fields.shape[0] != weight_index.size:
        raise ValueError(
            f"sensitivity_fields has {sensitivity_fields.shape[0]} features, "
            f"but the library has {weight_index.size} output features."
        )

    ratios: list[float] = []
    for k in range(int(library.K)):
        support = np.ix_(*library.inds_k[k])
        phi_dot = np.asarray(library.fulltweights[k], dtype=float)
        denominator = float(np.sum(phi_dot**2))
        if denominator <= 0.0:
            continue
        numerator_field = np.zeros(phi_dot.shape, dtype=float)
        for feature, index in enumerate(weight_index):
            if not np.any(sensitivity_fields[feature]):
                # A feature the model does not use, which is most of them for a
                # sparse system: skip rather than add zeros grid-point by point.
                continue
            weights = (
                library.fullweights0[k]
                if index < 0
                else library.fullweights1[k][index]
            )
            numerator_field += np.abs(np.asarray(weights, dtype=float)) * (
                sensitivity_fields[feature][support]
            )
        ratios.append(float(np.sum(numerator_field**2) / denominator))
    return np.asarray(ratios)



def pde_sensitivity_fields(library, data, coefficients, *, relative_step: float = 1e-4):
    """``|Xi_p| * |d(feature_p)/du|`` on the grid, for every output feature.

    The input :func:`pde_weak_validity_ratio` needs, built generically so a case
    does not have to differentiate its own library by hand. Each output feature
    block contributes differently:

    * plain library functions ``f_m(u)`` contribute ``|df_m/du|``, taken by a
      finite difference so any function library works, custom ones included;
    * pure integral terms are the raw state, so their sensitivity is 1;
    * mixed terms ``f_m(u) * u_n`` contribute by the product rule, bounded as
      ``|df_m/du| |u_n| + |f_m(u)|``.

    The coefficient enters as a magnitude: a feature the model does not use
    cannot invalidate the covariance model no matter how sharp it is, so a zero
    column contributes nothing.
    """

    data = np.asarray(data, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    strength = np.max(np.abs(coefficients), axis=0)

    weight_index = pde_feature_weight_index(library)
    n_output = weight_index.size
    if strength.size != n_output:
        raise ValueError(
            f"coefficients describe {strength.size} features, but the library "
            f"has {n_output} output features."
        )

    grid_shape = data.shape[:-1]
    n_features = data.shape[-1]

    def evaluate(values: np.ndarray) -> np.ndarray:
        arr = AxesArray(
            np.asarray(values, dtype=float),
            {"ax_spatial": list(range(len(grid_shape) - 1)),
             "ax_time": len(grid_shape) - 1,
             "ax_coord": len(grid_shape)},
        )
        return np.asarray(library.function_library.fit_transform(arr), dtype=float)

    funcs = evaluate(data)
    scale = float(np.std(data))
    step = relative_step * (scale if scale > 0.0 else 1.0)
    dfuncs = np.abs(evaluate(data + step) - funcs) / step
    n_library_terms = funcs.shape[-1]

    fields = np.zeros((n_output,) + grid_shape, dtype=float)
    cursor = 0
    if bool(library.include_bias):
        cursor += 1  # a constant cannot respond to the state
    for m in range(n_library_terms):
        fields[cursor] = strength[cursor] * dfuncs[..., m]
        cursor += 1
    for _ in range(int(library.num_derivatives)):
        for _n in range(n_features):
            fields[cursor] = strength[cursor]
            cursor += 1
    if bool(getattr(library, "include_interaction", True)):
        for _ in range(int(library.num_derivatives)):
            for n in range(n_features):
                for m in range(n_library_terms):
                    fields[cursor] = strength[cursor] * (
                        dfuncs[..., m] * np.abs(data[..., n]) + np.abs(funcs[..., m])
                    )
                    cursor += 1
    if cursor != n_output:
        raise ValueError(f"Filled {cursor} of {n_output} sensitivity fields.")
    return fields


def pde_scale_separation_ratio(
    H_t: float, H_x: float | Sequence[float], derivative_order: int
) -> float:
    """Support-width scaling of the PDE validity ratio, which needs no data.

    Superseded for reporting by :func:`pde_weak_validity_ratio`, which computes
    the ratio from the assembled system and is dimensionless. This one returns
    the bare monomial from an O(.) statement, so it carries units of
    time^2 / length^(2m) and its absolute value means nothing on its own --
    only how it scales as the supports shrink.

    For PDEs the neglected term is controlled by scale separation between the
    temporal and spatial supports rather than by the Jacobian: the ratio behaves
    as ``h_t^2 * h_x^(-2m)`` for spatial derivative order ``m``. The covariance
    model requires the temporal support to shrink sufficiently faster than the
    spatial ones.
    """

    widths = np.atleast_1d(np.asarray(H_x, dtype=float))
    if H_t <= 0 or np.any(widths <= 0):
        raise ValueError("Support widths must be positive.")
    return float(H_t**2 * np.min(widths) ** (-2 * int(derivative_order)))


def weak_design_report(library, K_requested: int | None = None) -> dict:
    """What a built library's weak design actually came out as.

    The requested K is not the K that gets used: it is clamped to the rank the
    design supports and then stripped of duplicate supports. Reporting the
    request would misstate the experiment -- at 100 Lorenz samples a request of
    100 test functions is fitted with 43 -- so the realised numbers are read back
    off the library. ``K_used`` depends on the placement draw, so it varies by a
    few between Monte Carlo seeds.
    """

    report = {
        "K_requested": None if K_requested is None else int(K_requested),
        "K_used": int(library.K),
        "duplicate_supports_dropped": int(getattr(library, "n_duplicate_domains_", 0)),
    }
    for attribute, name in (
        ("cov_cond_", "cov_cond"),
        ("cov_rank_", "cov_rank"),
        ("cov_rank_deficit_", "cov_rank_deficit"),
        ("cov_below_nugget_", "cov_below_nugget"),
    ):
        value = getattr(library, attribute, None)
        if value is not None:
            report[name] = float(value) if "cond" in name else int(value)
    return report
class WeightedWeakPDELibrary(DedupedWeakPDELibrary):
    """
    WeakPDELibrary with GLS whitening via a Cholesky factor built from the
    variance field on the spatiotemporal grid.

    Notes
    -----
    The whitener W = L^{-1}, with L L^T = Cov[V], is left-applied to both Θ and V.
    This implements min_x || W(Θ x - V) ||_2^2, i.e., GLS in the weak space.
    """

    def __init__(self, *args, spatiotemporal_weights=None, whitener_mode="full", **kwargs):
        if whitener_mode not in ("full", "diag"):
            raise ValueError(
                f"whitener_mode must be 'full' or 'diag', got {whitener_mode!r}."
            )
        self.spatiotemporal_weights = spatiotemporal_weights
        self.whitener_mode = whitener_mode
        self._L_chol = None  # lower-triangular Cholesky factor of Cov[V]
        self.cov_size_ = None
        self.cov_cond_ = None
        self.cov_rank_ = None
        self.cov_rank_deficit_ = None
        self.cov_below_nugget_ = None
        super().__init__(*args, **kwargs)

    # ------------------------------ core whitening ------------------------------

    def _build_whitener_from_variance(self):
        """
        Construct L such that Cov[V] = L L^T with
        Cov[V]_{kℓ} = sum_g w_k[g] w_ℓ[g] σ^2[g].
        """
        if self.spatiotemporal_weights is None:
            self._L_chol = None
            return

        base_grid = np.asarray(self.spatiotemporal_grid)
        expected = tuple(base_grid.shape[:-1])
        var_grid = np.asarray(self.spatiotemporal_weights)

        if var_grid.shape == expected + (1,):
            var_grid = var_grid[..., 0]
        elif var_grid.shape != expected:
            raise ValueError(
                f"spatiotemporal_weights must have shape {expected} or {expected + (1,)}, "
                f"got {var_grid.shape}"
            )

        var_flat = var_grid.ravel(order="C")
        sqrt_var_flat = np.sqrt(var_flat)
        grid_shape = expected
        K = self.K

        idx_lists = []
        val_lists = []

        for k in range(K):
            inds_axes = [np.asarray(ax, dtype=np.intp) for ax in self.inds_k[k]]
            grids = np.meshgrid(*inds_axes, indexing="ij")
            lin_idx = np.ravel_multi_index(tuple(grids), dims=grid_shape, order="C")
            lin_idx = lin_idx.ravel(order="C")

            wk = np.asarray(self.fulltweights[k], dtype=float).ravel(order="C")
            if wk.shape[0] != lin_idx.shape[0]:
                raise RuntimeError(
                    f"Weight/variance size mismatch on cell {k}: "
                    f"wk has {wk.shape[0]} entries, indices have {lin_idx.shape[0]}"
                )

            vals = wk * sqrt_var_flat[lin_idx]

            idx_lists.append(lin_idx)
            val_lists.append(vals)

        # --- Build sparse B and Cov = B B^T --------------------------------
        from scipy.sparse import csr_matrix

        G = var_flat.size
        data = np.concatenate(val_lists)
        indices = np.concatenate(idx_lists)
        indptr = np.zeros(K + 1, dtype=int)
        offset = 0
        for k in range(K):
            length = len(val_lists[k])
            indptr[k] = offset
            offset += length
        indptr[K] = offset

        B = csr_matrix((data, indices, indptr), shape=(K, G))
        Cov = (B @ B.T).toarray()  # K x K dense covariance

        # Nugget for numerical stability
        avg_diag = np.trace(Cov) / max(K, 1)
        nugget = 1e-12 * avg_diag
        Cov.flat[:: K + 1] += nugget

        # The nugget lets the Cholesky succeed on a covariance that is singular,
        # and the whitener then amplifies those directions by 1/sqrt(nugget).
        # Record the conditioning so a degenerate Sigma is visible rather than
        # silent: the domain centres are placed at random, so two of them landing
        # within one grid step give duplicate rows, and K too large for the grid
        # makes that the rule rather than the exception.
        eigenvalues = np.clip(np.linalg.eigvalsh(Cov)[::-1], 0.0, None)
        self.cov_size_ = int(K)
        self.cov_cond_ = float(np.linalg.cond(Cov))
        self.cov_rank_ = int(np.count_nonzero(eigenvalues > 1e-10 * eigenvalues[0]))
        self.cov_rank_deficit_ = int(K - self.cov_rank_)
        self.cov_below_nugget_ = int(np.count_nonzero(eigenvalues < nugget))

        if self.whitener_mode == "diag":
            # Variance-only weighting: keep the marginal weak variances, discard
            # the correlations induced by overlapping test-function supports.
            # A diagonal whitener cannot amplify a near-null direction, so the
            # conditioning recorded above does not apply to it.
            self._L_chol = np.diag(np.sqrt(np.diag(Cov)))
            return

        if self.cov_rank_deficit_ > 0:
            warnings.warn(
                f"Weak covariance is rank {self.cov_rank_} of {K} "
                f"(cond {self.cov_cond_:.2e}); whitening will amplify "
                f"{self.cov_rank_deficit_} direction(s) that carry no data. "
                "Reduce K, widen the test-function support, or sample the grid "
                "more finely.",
                WeakCovarianceWarning,
                stacklevel=2,
            )

        try:
            self._L_chol = np.linalg.cholesky(Cov)
        except np.linalg.LinAlgError:
            Cov.flat[:: K + 1] += max(1e-10, 1e-6 * avg_diag)
            self._L_chol = np.linalg.cholesky(Cov)


    def _apply_whitener(self, A):
        """Return L^{-1} A without forming L^{-1} explicitly."""
        if self._L_chol is None:
            return A
        return np.linalg.solve(self._L_chol, A)

    # ------------------------------ hooks ------------------------------

    def _weak_form_setup(self):
        # parent builds inds_k and the weak weight tensors
        super()._weak_form_setup()
        # then build the GLS whitener from the variance field
        if self.spatiotemporal_weights is not None:
            self._build_whitener_from_variance()

    def convert_u_dot_integral(self, u):
        Vy = super().convert_u_dot_integral(u)  # (K, 1)
        Vy_w = self._apply_whitener(np.asarray(Vy))
        return AxesArray(Vy_w, {"ax_sample": 0, "ax_coord": 1})

    def transform(self, x_full):
        VTheta_list = super().transform(x_full)  # list of (K, n_features)
        if self._L_chol is None:
            return VTheta_list
        out = []
        for VTheta in VTheta_list:
            A = np.asarray(VTheta)
            A_w = self._apply_whitener(A)  # (K, m)
            out.append(AxesArray(A_w, {"ax_sample": 0, "ax_coord": 1}))
        return out
