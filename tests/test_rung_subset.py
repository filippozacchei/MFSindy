"""Per-rung tuning is only sound if a rung fitted alone equals the same rung
fitted alongside the others: each rung must be reported under its own
hyperparameters and nothing else. The ensemble optimizer bags from the global
RNG, so this holds only while every rung's fit is seeded from its own stream.
"""

import numpy as np
import pytest

from mfsindy.cases.lorenz import (
    LorenzMultiTrajectoryGLSConfig,
    _lorenz_batch,
    _lorenz_fit_multi_trajectory_weak_gls_models,
    _lorenz_reference_state_std,
)
from mfsindy.experiments import PART1_METHODS


@pytest.fixture(scope="module")
def lorenz_fits():
    cfg = LorenzMultiTrajectoryGLSConfig(
        n_hf=1,
        n_lf=2,
        T_train=1.0,
        dt=0.01,
        noise_hf_rel=0.01,
        noise_lf_rel=0.1,
        n_ensemble_models=20,
        stlsq_threshold=0.1,
        H_xt=0.1,
    )
    state_std = _lorenz_reference_state_std(cfg)
    noise_hf = cfg.noise_hf_rel * state_std
    noise_lf = cfg.noise_lf_rel * state_std
    batch = _lorenz_batch(0, cfg, noise_hf, noise_lf)

    def fit(methods=None):
        return _lorenz_fit_multi_trajectory_weak_gls_models(
            batch,
            cfg,
            cfg.make_optimizer,
            t_argument=batch.t_argument,
            noise_hf_abs=noise_hf,
            noise_lf_abs=noise_lf,
            methods=methods,
        )

    return fit, fit()


def test_all_rungs_are_fitted_by_default(lorenz_fits):
    _, full = lorenz_fits
    assert set(full) == set(PART1_METHODS)


@pytest.mark.parametrize("rung", PART1_METHODS)
def test_rung_fitted_alone_matches_full_fit(lorenz_fits, rung):
    fit, full = lorenz_fits
    alone = fit(methods=[rung])
    assert set(alone) == {rung}
    np.testing.assert_array_equal(alone[rung], full[rung])


def test_unknown_rung_is_rejected(lorenz_fits):
    fit, _ = lorenz_fits
    with pytest.raises(KeyError, match="No block recipe"):
        fit(methods=["NOPE"])
