"""Part II's PDE cases build their four weightings through one shared builder.

Before it, burgers and the flow each carried their own copy of the same four
library constructions, and each pinned K to a constant -- 100 and 1000 -- so a
support could not be varied without silently varying the coverage with it. The
builder resolves K from the support instead, and places every rung on the same
test functions, which is what makes the rungs comparable at all.
"""

import warnings

import numpy as np
import pytest

from mfsindy.cases.burgers import BurgersConfig, _make_burgers_gls_artifacts
from mfsindy.experiments.artifacts import (
    build_pde_intra_trajectory_artifacts,
    pde_coverage_test_function_count,
)
from mfsindy.weighted_weak_pde_library import WeakCovarianceWarning

METHODS = ["No weighting", "Ones GLS", "Variance diag", "Variance GLS"]


#: The grid and viscosity these properties are stated for, pinned so the tests
#: do not follow the experiment's defaults while those are being tuned.
TEST_SETTINGS = dict(NX=256, nu=0.1, t1=10.0, n_traj=1)


def burgers_artifacts(**overrides):
    cfg = BurgersConfig(n_runs=1, **{**TEST_SETTINGS, **overrides})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WeakCovarianceWarning)
        return cfg, _make_burgers_gls_artifacts(
            0, cfg, np.random.default_rng(cfg.seed_base)
        )[0]


def test_coverage_count_is_the_volume_ratio():
    """K at coverage 2 is 2 * (grid volume) / (support volume)."""

    x = np.linspace(0.0, 4.0, 41)
    t = np.linspace(0.0, 2.0, 21)
    X, T = np.meshgrid(x, t)
    grid = np.asarray([X, T]).T                      # (nx, nt, 2), extents 4 and 2

    # supports of half-width 0.5 and 0.25 occupy 1.0 x 0.5 each
    assert pde_coverage_test_function_count(grid, [0.5, 0.25]) == round(2 * 8 / 0.5)
    # a scalar support applies to every dimension
    assert pde_coverage_test_function_count(grid, 0.5) == round(2 * 8 / 1.0)
    # halving the support in one dimension doubles the count
    wide = pde_coverage_test_function_count(grid, [1.0, 0.25])
    assert pde_coverage_test_function_count(grid, [0.5, 0.25]) == 2 * wide


def test_support_widths_must_be_positive():
    grid = np.zeros((4, 4, 2))
    grid[..., 0], grid[..., 1] = np.meshgrid(np.arange(4.0), np.arange(4.0))
    with pytest.raises(ValueError):
        pde_coverage_test_function_count(grid, [1.0, 0.0])


def test_every_rung_sits_on_the_same_test_functions():
    """The rungs differ by covariance model alone.

    If the four libraries drew their own domains, a difference between rungs
    would partly be a difference between placements, and the Part II comparison
    would not mean what it says.
    """

    _, artifacts = burgers_artifacts(K=40, H_xt=[2.0, 1.0])
    reference = artifacts.libraries["No weighting"]
    for method in METHODS:
        library = artifacts.libraries[method]
        assert library.K == reference.K
        for k in range(int(reference.K)):
            for axis, indices in enumerate(reference.inds_k[k]):
                assert np.array_equal(indices, library.inds_k[k][axis])


def test_K_follows_the_support_when_not_pinned():
    cfg, artifacts = burgers_artifacts(K=None, H_xt=[2.0, 1.0])
    library = artifacts.libraries["Variance GLS"]
    library.fit_transform([artifacts.data])

    grid_extents = 2.0 * cfg.L * (cfg.t1 - cfg.t0)
    expected = max(2, round(2.0 * grid_extents / (4.0 * 2.0)))
    assert int(library.K) == expected
    assert library.cov_rank_deficit_ == 0

    # and halving the support in time roughly doubles the count
    _, finer = burgers_artifacts(K=None, H_xt=[2.0, 0.5])
    assert int(finer.libraries["Variance GLS"].K) == 2 * expected


def test_pinned_K_is_still_honoured():
    """Existing configs that name a K must keep getting it."""

    _, artifacts = burgers_artifacts(K=100, H_xt=None)
    assert all(int(artifacts.libraries[m].K) == 100 for m in METHODS)


def test_one_of_function_library_or_poly_degree_is_required():
    grid = np.zeros((4, 4, 2))
    grid[..., 0], grid[..., 1] = np.meshgrid(np.arange(4.0), np.arange(4.0))
    data = np.zeros((4, 4, 1))
    kwargs = dict(
        spatiotemporal_grid=grid,
        t_grid=np.arange(4.0),
        variance=np.ones((4, 4)),
        derivative_order=1,
        K=2,
        H_xt=[1.0, 1.0],
    )
    with pytest.raises(ValueError, match="exactly one"):
        build_pde_intra_trajectory_artifacts(data, **kwargs)
    with pytest.raises(ValueError, match="exactly one"):
        build_pde_intra_trajectory_artifacts(
            data, poly_degree=1, function_library=object(), **kwargs
        )


def test_constant_sigma_cancels_from_the_pde_kappa():
    """As for the ODE kappa: one definition serves both parts only if a noise
    level constant over the grid drops out, which is what lets Part I keep
    omitting it."""

    from mfsindy.cases.burgers import build_true_burgers_coefficients
    from mfsindy.weighted_weak_pde_library import (
        pde_sensitivity_fields,
        pde_weak_validity_ratio,
    )

    cfg, artifacts = burgers_artifacts(K=None, H_xt=[2.0, 1.0])
    library = artifacts.libraries["No weighting"]
    library.fit_transform([artifacts.data])
    sensitivity = pde_sensitivity_fields(
        library, artifacts.data, build_true_burgers_coefficients(cfg.nu)
    )
    grid_shape = artifacts.data.shape[:-1]

    unweighted = pde_weak_validity_ratio(library, sensitivity)
    for level in (1.0, 4.2, 1e-2):
        weighted = pde_weak_validity_ratio(
            library, sensitivity, sigma=np.full(grid_shape, level)
        )
        assert np.allclose(unweighted, weighted, rtol=0, atol=1e-12)


def test_pde_kappa_is_small_where_the_covariance_model_is_claimed_to_hold():
    """A sanity floor on the diagnostic itself: at a support well inside the
    grid the neglected term should be a small fraction of the retained one, or
    the ratio is measuring something other than what the appendix derives."""

    from mfsindy.cases.burgers import build_true_burgers_coefficients
    from mfsindy.weighted_weak_pde_library import (
        pde_sensitivity_fields,
        pde_weak_validity_ratio,
    )

    cfg, artifacts = burgers_artifacts(K=None, H_xt=[2.0, 1.0])
    library = artifacts.libraries["No weighting"]
    library.fit_transform([artifacts.data])
    ratios = pde_weak_validity_ratio(
        library,
        pde_sensitivity_fields(
            library, artifacts.data, build_true_burgers_coefficients(cfg.nu)
        ),
    )
    assert np.median(ratios) < 1.0
