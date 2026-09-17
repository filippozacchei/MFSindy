import warnings

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
    """WeakPDELibrary that discards duplicate test-function supports."""

    def _weak_form_setup(self):
        super()._weak_form_setup()
        self.n_duplicate_domains_ = drop_duplicate_domains(self)


#: Cached usable test-function counts, keyed by the weak design (grid, support,
#: polynomial degree, requested K). The count depends only on that design, not on
#: the data, so every trajectory and Monte Carlo seed reuses one measurement.
_USABLE_K_CACHE: dict = {}


def weak_design_rank(library) -> int:
    """How many of a built library's test functions give independent equations.

    The weak covariance is ``B B^T`` for the weight matrix ``B``, one row per test
    function, so ``rank(Sigma) = rank(B)`` and the redundancy can be measured
    before any data is fitted.

    Rows go dependent two ways. pysindy recentres each domain onto grid points,
    so two centres falling in the same interval produce byte-identical rows --
    for narrow supports the rank equals the number of distinct supports exactly.
    Wide supports lose further rank as overlapping bumps become near-dependent.
    Neither is predictable from a formula worth trusting, so this measures it.
    """

    K = int(library.K)
    grid_shape = tuple(np.asarray(library.spatiotemporal_grid).shape[:-1])
    n_grid = int(np.prod(grid_shape))

    B = np.zeros((K, n_grid), dtype=float)
    for k in range(K):
        axes = [np.asarray(ax, dtype=np.intp) for ax in library.inds_k[k]]
        mesh = np.meshgrid(*axes, indexing="ij")
        flat = np.ravel_multi_index(tuple(mesh), dims=grid_shape, order="C").ravel(order="C")
        B[k, flat] = np.asarray(library.fulltweights[k], dtype=float).ravel(order="C")

    singular_values = np.linalg.svd(B, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return 0
    return int(np.count_nonzero(singular_values > 1e-10 * singular_values[0]))


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


def usable_test_functions(build_probe, design_key, K_requested: int) -> int:
    """Clamp a requested test-function count to the number that is usable.

    Asking for more test functions than the weak design supports does not add
    information: the surplus equations are linear combinations of the others,
    and whitening by a covariance with those directions in it divides by
    round-off, amplifying noise by ~1e6. Clamping keeps the request honest.

    ``build_probe(K)`` must return a built library at that count; it is called at
    most once per distinct ``design_key``.
    """

    K_requested = int(K_requested)
    key = (design_key, K_requested)
    if key not in _USABLE_K_CACHE:
        # Building the probe places domains, which draws from the global RNG.
        # Left alone it would shift the placement of the library built next, and
        # only on a cache miss -- so the same configuration would give different
        # test functions depending on whether it had been measured before.
        rng_state = np.random.get_state()
        try:
            rank = weak_design_rank(build_probe(K_requested))
        finally:
            np.random.set_state(rng_state)
        _USABLE_K_CACHE[key] = max(1, min(K_requested, rank))
    return _USABLE_K_CACHE[key]


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
