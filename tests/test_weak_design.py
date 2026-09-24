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
    _lorenz_jacobian_norm,
    _lorenz_reference_state_std,
)
from mfsindy.weighted_weak_pde_library import (
    WeakCovarianceWarning,
    weak_validity_ratio,
)


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


def test_keeping_duplicates_makes_the_covariance_singular():
    """Deduplication is the whole of the fix, so turning it off must bring the
    singular covariance back.

    K is set explicitly rather than left to the support rule. At coverage 1 the
    rule asks for few enough test functions that the design is benign either way,
    so a test resting on the default would pass for the wrong reason and stop
    testing anything the day the rule changed.
    """
    requested = 200

    def build(deduplicate):
        cfg = LorenzMultiTrajectoryGLSConfig(
            n_hf=1, n_lf=2, H_xt=0.05, dt=0.01, noise_lf_rel=0.25, K=requested
        )
        cfg.deduplicate = deduplicate
        state_std = _lorenz_reference_state_std(cfg)
        noise_hf = cfg.noise_hf_rel * state_std
        batch = _lorenz_batch(0, cfg, noise_hf, cfg.noise_lf_rel * state_std)
        variance = np.full(batch.hf[0].shape[:-1], noise_hf**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WeakCovarianceWarning)
            library = _lorenz_make_weak_library(batch, cfg, variance_field=variance)
            library.fit_transform([batch.hf[0]])
        return library

    kept = build(False)
    assert kept.K == requested                      # every request kept
    assert kept.n_duplicate_domains_ == 0           # including the copies
    assert kept.cov_rank_deficit_ > 0               # so the covariance is singular

    deduped = build(True)
    assert deduped.K < requested
    assert deduped.n_duplicate_domains_ > 0
    assert deduped.cov_rank_deficit_ == 0


def test_deduplication_defaults_to_on():
    assert LorenzMultiTrajectoryGLSConfig().deduplicate is True


def build_library_and_trajectory(H_xt=0.05, dt=0.01):
    """As :func:`build_library`, but hands back the trajectory kappa is measured on."""

    cfg = LorenzMultiTrajectoryGLSConfig(
        n_hf=1, n_lf=2, H_xt=H_xt, dt=dt, K=None, noise_lf_rel=0.25
    )
    state_std = _lorenz_reference_state_std(cfg)
    noise_hf = cfg.noise_hf_rel * state_std
    batch = _lorenz_batch(0, cfg, noise_hf, cfg.noise_lf_rel * state_std)
    trajectory = batch.hf[0]
    variance = np.full(trajectory.shape[:-1], noise_hf**2)
    library = _lorenz_make_weak_library(batch, cfg, variance_field=variance)
    library.fit_transform([trajectory])
    return library, trajectory


def test_constant_sigma_cancels_from_kappa():
    """Part I passes no sigma because its noise is constant along a trajectory.

    That is only sound if a constant sigma cancels between the two terms of the
    ratio, which is what lets one definition serve both parts: Part I's numbers
    must not move now that the argument exists.
    """
    library, trajectory = build_library_and_trajectory()
    jacobian = _lorenz_jacobian_norm(trajectory)
    n_points = trajectory.shape[0]

    unweighted = weak_validity_ratio(library, jacobian)
    for level in (1.0, 3.7, 1e-3):
        weighted = weak_validity_ratio(
            library, jacobian, sigma=np.full(n_points, level)
        )
        assert np.allclose(unweighted, weighted, rtol=0, atol=1e-12)


def test_varying_sigma_reweights_kappa():
    """Under heteroscedastic noise sigma does not cancel, which is the whole
    reason Part II carries it: the ratio is then dominated by whichever stretch
    of the trajectory is noisiest.

    A sigma concentrated on one half of the trajectory must pull kappa towards
    the test functions living there, so the two halves cannot agree.
    """
    library, trajectory = build_library_and_trajectory()
    jacobian = _lorenz_jacobian_norm(trajectory)
    n_points = trajectory.shape[0]

    early = np.concatenate([np.ones(n_points // 2), np.full(n_points - n_points // 2, 1e-3)])
    late = early[::-1].copy()

    kappa_early = weak_validity_ratio(library, jacobian, sigma=early)
    kappa_late = weak_validity_ratio(library, jacobian, sigma=late)
    assert not np.allclose(kappa_early, kappa_late)
