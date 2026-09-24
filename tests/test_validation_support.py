"""The validation weak system is a choice, and it has to be made deliberately.

Every candidate in the tuning grid is scored against one fixed weak system, so
that system's support width decides what the scores mean. Two bounds pin it
down from opposite ends: too narrow and noise in ``b = \\int \\dot\\phi u`` caps
the attainable score, too wide and the target itself breaks the covariance
model the appendix relies on.
"""

import warnings

import numpy as np
import pytest

from mfsindy.cases.lorenz import (
    LorenzMultiTrajectoryGLSConfig,
    build_lorenz_weak_validation_blocks,
    build_true_coefficient_matrix,
    generate_lorenz_dataset,
    lorenz_validation_support,
)
from mfsindy.experiments import evaluate_weak_form_models

CANDIDATES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


@pytest.fixture(scope="module")
def draw():
    cfg = LorenzMultiTrajectoryGLSConfig(dt=1e-3, T_train=1.0)
    sigma = cfg.noise_hf_rel * 13.636366644425403  # reference state std
    traj, t_grid, _ = generate_lorenz_dataset(
        n_traj=1, T=cfg.T_train, dt=cfg.dt, noise_level=sigma, seed=4242
    )
    return cfg, traj, t_grid, sigma


def test_rule_binds_on_the_noise_ceiling(draw):
    cfg, traj, t_grid, sigma = draw
    support = lorenz_validation_support(
        cfg, traj[0], t_grid=t_grid, sigma=sigma, candidates=CANDIDATES, weak_seed=11
    )
    table = support.table.set_index("H_val")
    # Narrow supports fail: b inherits the noise through ||phi_dot||.
    assert not table.loc[0.005, "ceiling_ok"]
    # The chosen width clears the ceiling and is the smallest that does, so it
    # carries the most rows of any qualifying width.
    assert support.satisfied
    assert support.ceiling >= 0.99
    assert support.K == int(table[table["ceiling_ok"]]["K"].max())


def test_kappa_is_reported_but_does_not_bind(draw):
    """Kappa governs the covariance model, which this system never invokes.

    The validation library is built unweighted and scored with an unweighted
    R^2, so a wide support with a large kappa is still a legitimate target. It
    stays in the table as context for the rungs that do whiten.
    """

    cfg, traj, t_grid, sigma = draw
    support = lorenz_validation_support(
        cfg, traj[0], t_grid=t_grid, sigma=sigma, candidates=CANDIDATES, weak_seed=11
    )
    assert support.table["kappa"].notna().all()
    # With no bound set there is no gating column to mistake for a check that
    # was made and passed.
    assert "kappa_ok" not in support.table.columns
    assert "rows_ok" not in support.table.columns


def test_analytic_ceiling_matches_the_true_model(draw):
    """The ceiling is predicted without the true coefficients; check it holds."""

    cfg, traj, t_grid, sigma = draw
    support = lorenz_validation_support(
        cfg, traj[0], t_grid=t_grid, sigma=sigma, candidates=CANDIDATES, weak_seed=11
    )
    blocks = build_lorenz_weak_validation_blocks(
        cfg, traj, t_grid=t_grid, H_val=support.H_val, weak_seed=11
    )
    scored = evaluate_weak_form_models(
        {"true": build_true_coefficient_matrix().T}, blocks
    )
    achieved = float(scored["value"].iloc[0])
    assert achieved == pytest.approx(support.ceiling, abs=2e-3)


def test_blocks_ignore_the_candidate_support(draw):
    """A fixed H_val must produce the same target whatever the config says."""

    cfg, traj, t_grid, _ = draw
    wide = build_lorenz_weak_validation_blocks(
        cfg, traj, t_grid=t_grid, H_val=0.05, weak_seed=11
    )[0]
    from dataclasses import replace

    narrow_cfg = replace(cfg, H_xt=0.005, K=None)
    same = build_lorenz_weak_validation_blocks(
        narrow_cfg, traj, t_grid=t_grid, H_val=0.05, weak_seed=11
    )[0]
    assert np.array_equal(wide.theta, same.theta)
    assert np.array_equal(wide.rhs, same.rhs)


def test_unsatisfiable_bounds_warn_rather_than_fail(draw):
    cfg, traj, t_grid, sigma = draw
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        support = lorenz_validation_support(
            cfg,
            traj[0],
            t_grid=t_grid,
            sigma=sigma,
            candidates=CANDIDATES,
            weak_seed=11,
            min_ceiling=0.999999999,
        )
    assert not support.satisfied
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert support.H_val in CANDIDATES


# ---------------------------------------------------------------------------
# Per-rung kappa gate
# ---------------------------------------------------------------------------


def test_kappa_gates_only_the_full_covariance_rungs():
    """VMF whitens, but kappa does not bound it.

    Diagonal whitening uses only the marginal variances, and GLS is invariant
    to a global rescale of its weights, so the level of kappa cancels: at a
    kappa of 3.2 on Lorenz its weights are off by 1.5x, not 4x. The gate is for
    the rungs that use the test-function correlations, where the error does not
    cancel -- and PMF uses nothing else.
    """

    from mfsindy.experiments import (
        FULL_COVARIANCE_RUNGS,
        RUNG_BLOCKS,
        WHITENING_RUNGS,
    )

    assert WHITENING_RUNGS == {"VHF", "VLF", "PMF", "VMF", "MF_w"}
    assert FULL_COVARIANCE_RUNGS == {"VHF", "VLF", "PMF", "MF_w"}
    assert "VMF" not in FULL_COVARIANCE_RUNGS
    assert all(w == "diag" for _, w in RUNG_BLOCKS["VMF"])
    for rung in ("HF", "LF", "MF"):
        assert all(w == "plain" for _, w in RUNG_BLOCKS[rung])


def test_kappa_table_is_deterministic_and_monotone(draw):
    from mfsindy.cases.lorenz import lorenz_kappa_by_support

    cfg, _, _, _ = draw
    first = lorenz_kappa_by_support(cfg, CANDIDATES, n_reference=2)
    second = lorenz_kappa_by_support(cfg, CANDIDATES, n_reference=2)
    # The gate must not move with the Monte Carlo draw.
    assert np.allclose(first["kappa_median"], second["kappa_median"])
    assert np.allclose(first["kappa_max"], second["kappa_max"])
    # Kappa grows with the support, so the admissible set is a prefix.
    assert np.all(np.diff(first["kappa_median"]) > 0)
    assert np.all(first["kappa_max"] >= first["kappa_median"])


def test_admissible_set_does_not_move_with_the_seed():
    """The gate statistic must not depend on which references were drawn.

    T/20 is the marginal support on Lorenz: individual trajectories put it
    anywhere from 0.17 to 0.44, straddling the 0.25 bound. The median over
    references is stable where the worst case is not, which is why the gate
    uses it -- a bound that is a guideline should not let a seed change decide
    the grid. The reported range is what carries the marginality.
    """

    from dataclasses import replace

    from mfsindy.cases.lorenz import lorenz_kappa_by_support

    base = LorenzMultiTrajectoryGLSConfig(p=2)
    sets = []
    for seed in (0, 1234, 77):
        table = lorenz_kappa_by_support(replace(base, seed_base=seed), CANDIDATES)
        sets.append(frozenset(table.loc[table["kappa_median"] <= 0.25, "H_xt"]))
        # The range must bracket the gate statistic, and be materially wide at
        # T/20 -- that width is the marginality the table is there to show.
        row = table.set_index("H_xt").loc[0.05]
        assert row["kappa_lo"] <= row["kappa_median"] <= row["kappa_hi"]
        assert row["kappa_hi"] / row["kappa_lo"] > 1.3
    assert len(set(sets)) == 1, f"admissible set moved with the seed: {sets}"
    assert sets[0] == frozenset({0.005, 0.01, 0.02, 0.05})


def test_gate_restricts_only_the_whitening_rungs():
    import pandas as pd

    from mfsindy.experiments import FULL_COVARIANCE_RUNGS, tune_rungs

    grid = {"H_xt": [0.01, 0.02, 0.2]}
    rungs = ["MF", "MF_w"]
    # A surface whose best cell is the widest support, for every rung.
    def evaluate(candidate):
        return pd.DataFrame(
            [
                {"model": r, "metric": "weak_r2", "value": float(candidate.H_xt)}
                for r in rungs
            ]
        )

    def admissible(rung, params):
        return rung not in FULL_COVARIANCE_RUNGS or params["H_xt"] <= 0.02

    cfg = LorenzMultiTrajectoryGLSConfig()
    selections, table = tune_rungs(
        cfg, param_grid=grid, rungs=rungs, evaluate=evaluate,
        metric="weak_r2", admissible=admissible,
    )
    # The unweighted rung keeps the whole grid and takes the widest support.
    assert selections["MF"].best_params["H_xt"] == 0.2
    assert not selections["MF"].restricted
    # The weighted one is held back, and the price is recorded.
    assert selections["MF_w"].best_params["H_xt"] == 0.02
    assert selections["MF_w"].restricted
    assert selections["MF_w"].unrestricted_best_params["H_xt"] == 0.2
    # Every cell is still scored, so the surface stays complete for the plot.
    assert len(table) == len(grid["H_xt"]) * len(rungs)
    assert not table["admissible"].all()


def test_vector_valued_grid_params_survive_selection():
    """The PDE benchmarks tune a vector support, so grid values are lists.

    Comparing a column of lists against one list with pandas ``==`` broadcasts
    elementwise and raises, which is not the question being asked. The boundary
    flags and the restricted-grid bookkeeping both do that comparison.
    """

    import pandas as pd

    from mfsindy.experiments import FULL_COVARIANCE_RUNGS, tune_rungs

    grid = {"H_xt": [[0.4, 0.05], [0.4, 0.1], [0.8, 0.1]]}
    rungs = ["MF", "MF_w"]

    def evaluate(candidate):
        return pd.DataFrame(
            [
                {"model": r, "metric": "weak_r2", "value": float(sum(candidate.H_xt))}
                for r in rungs
            ]
        )

    def admissible(rung, params):
        return rung not in FULL_COVARIANCE_RUNGS or params["H_xt"] != [0.8, 0.1]

    selections, table = tune_rungs(
        LorenzMultiTrajectoryGLSConfig(), param_grid=grid, rungs=rungs,
        evaluate=evaluate, metric="weak_r2", admissible=admissible,
    )
    assert selections["MF"].best_params["H_xt"] == [0.8, 0.1]
    assert selections["MF_w"].best_params["H_xt"] == [0.4, 0.1]
    assert selections["MF_w"].restricted
    # MF_w now sits at the top of the grid it can actually use, not the one it
    # was handed, so it must not be flagged as pinned against the full grid.
    assert selections["MF_w"].at_boundary.get("H_xt") != "high"


def test_group_library_cache_is_bit_identical():
    """One library per group must give exactly what one per trajectory gave.

    The cached parts -- domain placement, quadrature weights, covariance
    Cholesky -- depend only on things a group holds fixed, so reuse is a pure
    saving. Anything else here is a silent change to published numbers.
    """

    import mfsindy.experiments.multi_trajectory as M
    from mfsindy.cases.lorenz import (
        _lorenz_batch,
        _lorenz_fit_multi_trajectory_weak_gls_models,
        _lorenz_reference_state_std,
    )

    cfg = LorenzMultiTrajectoryGLSConfig(n_hf=2, n_lf=3, H_xt=0.1, n_ensemble_models=4)
    std = _lorenz_reference_state_std(cfg)
    hf_abs, lf_abs = cfg.noise_hf_rel * std, cfg.noise_lf_rel * std
    batch = _lorenz_batch(0, cfg, hf_abs, lf_abs)

    cached = _lorenz_fit_multi_trajectory_weak_gls_models(
        batch, cfg, cfg.make_optimizer, t_argument=cfg.dt,
        noise_hf_abs=hf_abs, noise_lf_abs=lf_abs,
    )
    # Defeat the cache: a signature of None is never shared.
    original = M._variance_signature
    M._variance_signature = lambda field: None
    try:
        per_trajectory = _lorenz_fit_multi_trajectory_weak_gls_models(
            batch, cfg, cfg.make_optimizer, t_argument=cfg.dt,
            noise_hf_abs=hf_abs, noise_lf_abs=lf_abs,
        )
    finally:
        M._variance_signature = original

    assert set(cached) == set(per_trajectory)
    for rung in cached:
        np.testing.assert_array_equal(
            cached[rung], per_trajectory[rung],
            err_msg=f"library caching changed the {rung} coefficients",
        )


def test_heteroscedastic_variance_is_never_shared():
    """A field that varies sample to sample needs its own whitener."""

    from mfsindy.experiments.multi_trajectory import _variance_signature

    assert _variance_signature(None) == ("plain",)
    flat = np.full((4, 3), 0.25)
    assert _variance_signature(flat) == ((4, 3), 0.25)
    varying = flat.copy()
    varying[0, 0] = 0.5
    assert _variance_signature(varying) is None


# ---------------------------------------------------------------------------
# The isothermal benchmark scores against the equations, not against a fit
# ---------------------------------------------------------------------------


def test_analytic_ns_coefficients_match_the_solver():
    """Every term of ``compressible``, with the coefficients it actually uses.

    The old reference was a weak-SINDy fit of the clean flow, which made the
    benchmark circular and was wrong besides. These come from the RHS itself,
    so they are checked against it term by term rather than against another fit.
    """

    from mfsindy.cases.isothermal_flow import (
        NSIsothermalMultiTrajectoryGLSConfig,
        build_true_ns_isothermal_coefficients,
    )

    cfg = NSIsothermalMultiTrajectoryGLSConfig(RT=2.0, mu=3.0)
    names = [
        "uu_1", "vu_2", "rho^-1rho_1", "rho^-1u_11", "rho^-1u_22",
        "uv_1", "vv_2", "rho^-1rho_2", "rho^-1v_11", "rho^-1v_22",
        "urho_1", "vrho_2", "rhou_1", "rhov_2", "spurious",
    ]
    C = build_true_ns_isothermal_coefficients(names, RT=cfg.RT, mu=cfg.mu)
    at = lambda state, name: C[state, names.index(name)]

    # u_t = -u u_x - v u_y - RT rho^-1 rho_x + mu rho^-1 (u_xx + u_yy)
    assert at(0, "uu_1") == -1.0 and at(0, "vu_2") == -1.0
    assert at(0, "rho^-1rho_1") == -2.0            # -RT
    assert at(0, "rho^-1u_11") == 3.0 and at(0, "rho^-1u_22") == 3.0   # mu
    # v_t is the same equation with the axes swapped
    assert at(1, "uv_1") == -1.0 and at(1, "vv_2") == -1.0
    assert at(1, "rho^-1rho_2") == -2.0
    assert at(1, "rho^-1v_11") == 3.0 and at(1, "rho^-1v_22") == 3.0
    # rho_t = -(u rho)_x - (v rho)_y, expanded
    for name in ("urho_1", "vrho_2", "rhou_1", "rhov_2"):
        assert at(2, name) == -1.0
    # a leading library name is the FUNCTION factor: rhou_1 is rho*u_x while
    # urho_1 is u*rho_x. Both are real, and easy to transpose.
    assert at(0, "rhou_1") == 0.0 and at(1, "rhou_1") == 0.0
    # nothing leaks into features the equations do not use
    assert at(0, "spurious") == at(1, "spurious") == at(2, "spurious") == 0.0
    assert [int((np.abs(C[s]) > 0).sum()) for s in range(3)] == [5, 5, 4]


def test_missing_library_term_is_not_silently_dropped():
    """A library that cannot express the physics must fail, not truncate truth."""

    from mfsindy.cases.isothermal_flow import build_true_ns_isothermal_coefficients

    with pytest.raises(KeyError, match="rho\\^-1u_22"):
        build_true_ns_isothermal_coefficients(
            ["uu_1", "vu_2", "rho^-1rho_1", "rho^-1u_11"], RT=1.0, mu=1.0
        )


# ---------------------------------------------------------------------------
# Selection: one standard error, then parsimony
# ---------------------------------------------------------------------------


def _surface(rows):
    """Build an evaluate() from (H_xt, take-scores, n_terms) triples."""

    import pandas as pd

    def evaluate(candidate):
        scores, n_terms = rows[candidate.H_xt]
        out = []
        for s in scores:
            out.append({"model": "MF_w", "metric": "weak_r2", "value": s})
            out.append({"model": "MF_w", "metric": "n_active_terms", "value": n_terms})
        return pd.DataFrame(out)

    return evaluate


def test_parsimony_breaks_ties_the_score_cannot_separate():
    """The isothermal case in miniature: the sparser model loses by 1.4e-4.

    Weak R^2 weights a term by the signal it carries, so on that benchmark a
    true term is worth 4e-4 to delete -- less than the spread between grid
    cells. Ranking on score alone then prefers the model with spurious terms.
    """

    from mfsindy.experiments import tune_rungs

    # Take-to-take spread (different test-function placements) comparable to
    # the gap between the cells: the regime where the ranking means nothing.
    rows = {
        0.05: ([0.9994, 0.9990, 0.9999, 0.9997, 0.9992], 14.0),   # correct, sparse
        0.02: ([0.9995, 0.9991, 1.0000, 0.9998, 0.9993], 19.0),   # spurious terms
    }
    selections, table = tune_rungs(
        LorenzMultiTrajectoryGLSConfig(), param_grid={"H_xt": [0.02, 0.05]},
        rungs=["MF_w"], evaluate=_surface(rows), metric="weak_r2",
    )
    sel = selections["MF_w"]
    # The argmax wants the denser model; the rule takes the sparser one.
    assert sel.argmax_params == {"H_xt": 0.02}
    assert sel.best_params == {"H_xt": 0.05}
    assert sel.n_terms == 14.0
    assert sel.score_sem > 0.0
    assert "n_terms" in table.columns and "score_sem" in table.columns


def test_a_genuinely_worse_cell_never_wins_on_sparsity():
    """Parsimony only arbitrates ties, so it cannot rescue a bad candidate."""

    from mfsindy.experiments import tune_rungs

    rows = {
        0.05: ([0.999, 0.999, 0.999, 0.999, 0.999], 14.0),
        0.20: ([0.2, 0.2, 0.2, 0.2, 0.2], 3.0),      # far worse, and far sparser
    }
    selections, _ = tune_rungs(
        LorenzMultiTrajectoryGLSConfig(), param_grid={"H_xt": [0.05, 0.20]},
        rungs=["MF_w"], evaluate=_surface(rows), metric="weak_r2",
    )
    assert selections["MF_w"].best_params == {"H_xt": 0.05}
    assert selections["MF_w"].argmax_params is None   # score and rule agree


def test_rule_declines_when_the_score_really_does_separate():
    """If the takes agree far more closely than the cells differ, trust the score.

    One standard error then fails to span the gap and this reduces to the plain
    argmax -- the honest outcome, since the ranking is meaningful at that scale.
    """

    from mfsindy.experiments import tune_rungs

    rows = {
        0.05: ([0.99960, 0.99960, 0.99961, 0.99959, 0.99960], 14.0),
        0.02: ([0.99974, 0.99974, 0.99975, 0.99973, 0.99974], 19.0),
    }
    selections, _ = tune_rungs(
        LorenzMultiTrajectoryGLSConfig(), param_grid={"H_xt": [0.02, 0.05]},
        rungs=["MF_w"], evaluate=_surface(rows), metric="weak_r2",
    )
    assert selections["MF_w"].best_params == {"H_xt": 0.02}
    assert selections["MF_w"].argmax_params is None


def test_selection_falls_back_to_argmax_without_model_sizes():
    """An evaluate() that reports no term count still selects, on score alone."""

    import pandas as pd
    from mfsindy.experiments import tune_rungs

    def evaluate(candidate):
        return pd.DataFrame([{"model": "MF_w", "metric": "weak_r2",
                              "value": float(candidate.H_xt)}])

    selections, _ = tune_rungs(
        LorenzMultiTrajectoryGLSConfig(), param_grid={"H_xt": [0.05, 0.20]},
        rungs=["MF_w"], evaluate=evaluate, metric="weak_r2",
    )
    assert selections["MF_w"].best_params == {"H_xt": 0.20}
