"""The weak covariance must be invertible for whitening to mean anything.

pysindy recentres randomly placed domains onto grid points, so duplicate rows
appear long before K reaches the number of grid positions, and overlapping wide
supports add further dependence. Either makes the covariance singular, and the
whitener then divides a round-off direction by the nugget.
"""

import warnings

import numpy as np
import pytest

from mfsindy.cases.lorenz import (
    LorenzMultiTrajectoryGLSConfig,
    _lorenz_batch,
    _lorenz_make_weak_library,
    _lorenz_reference_state_std,
)
from mfsindy.weighted_weak_pde_library import WeakCovarianceWarning
import mfsindy.weighted_weak_pde_library as weak_module


def build_library(H_xt, dt=0.01, K=None):
    cfg = LorenzMultiTrajectoryGLSConfig(
        n_hf=1, n_lf=2, H_xt=H_xt, dt=dt, K=K, noise_lf_rel=0.25
    )
    state_std = _lorenz_reference_state_std(cfg)
    noise_hf = cfg.noise_hf_rel * state_std
    batch = _lorenz_batch(0, cfg, noise_hf, cfg.noise_lf_rel * state_std)
    variance = np.full(batch.hf[0].shape[:-1], noise_hf**2)
    library = _lorenz_make_weak_library(batch, cfg, variance_field=variance)
    library.fit_transform([batch.hf[0]])
    return library


@pytest.mark.parametrize("H_xt", [0.05, 0.1, 0.2])
def test_weak_covariance_is_full_rank(H_xt):
    library = build_library(H_xt)
    assert library.cov_rank_deficit_ == 0, (
        f"{library.cov_rank_deficit_} of {library.cov_size_} directions carry no "
        "variance; whitening would amplify them"
    )
    assert library.cov_below_nugget_ == 0


@pytest.mark.parametrize("H_xt", [0.05, 0.1, 0.2])
def test_no_warning_for_a_usable_covariance(H_xt):
    with warnings.catch_warnings():
        warnings.simplefilter("error", WeakCovarianceWarning)
        build_library(H_xt)


def test_duplicate_supports_are_dropped():
    # far more test functions than the grid can place distinctly
    library = build_library(0.05, K=200)
    assert library.n_duplicate_domains_ > 0
    supports = {
        tuple(np.asarray(library.inds_k[k][0]).ravel().tolist())
        for k in range(library.K)
    }
    assert len(supports) == library.K


def test_probe_does_not_disturb_the_placement():
    """The count is measured with a probe library, which draws domain centres.

    Left unrestored that draw would shift the placement of the library built
    next, and only when the cache missed -- so the same settings would give
    different test functions depending on whether they had been measured before.
    """
    weak_module._USABLE_K_CACHE.clear()
    cold = build_library(0.05)
    warm = build_library(0.05)
    assert cold.K == warm.K
    for k in range(cold.K):
        np.testing.assert_array_equal(
            np.asarray(cold.inds_k[k][0]), np.asarray(warm.inds_k[k][0])
        )


def test_clamp_false_restores_the_degenerate_design():
    """The flag exists to measure what the clamp is worth, so it must really
    disable it: no rank cap, no duplicate removal, and the singular covariance
    that follows."""
    clamped = build_library(0.05)
    cfg = LorenzMultiTrajectoryGLSConfig(
        n_hf=1, n_lf=2, H_xt=0.05, dt=0.01, noise_lf_rel=0.25
    )
    cfg.clamp = False
    state_std = _lorenz_reference_state_std(cfg)
    noise_hf = cfg.noise_hf_rel * state_std
    batch = _lorenz_batch(0, cfg, noise_hf, cfg.noise_lf_rel * state_std)
    variance = np.full(batch.hf[0].shape[:-1], noise_hf**2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WeakCovarianceWarning)
        unclamped = _lorenz_make_weak_library(batch, cfg, variance_field=variance)
        unclamped.fit_transform([batch.hf[0]])

    assert unclamped.n_duplicate_domains_ == 0      # nothing dropped
    assert unclamped.K > clamped.K                  # nothing capped
    assert unclamped.cov_rank_deficit_ > 0          # and the covariance is singular
    assert clamped.cov_rank_deficit_ == 0


def test_clamp_defaults_to_on():
    assert LorenzMultiTrajectoryGLSConfig().clamp is True
