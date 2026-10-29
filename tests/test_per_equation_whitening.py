"""Noise given per state component whitens each equation by its own factor.

Each equation's weak residual is driven by the noise on its own component, so
with a variance field per component the library builds one whitener per
equation. With the same variance on every component that must reproduce the
shared whitener exactly; with different variances each equation must follow its
own.
"""
import numpy as np
import pysindy as ps

from mfsindy.weighted_weak_pde_library import WeightedWeakPDELibrary, weak_system


def _library(variance, seed=0):
    t = np.linspace(0.0, 2.0, 401)
    np.random.seed(seed)
    return WeightedWeakPDELibrary(
        function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
        derivative_order=1, spatiotemporal_grid=t[:, None], is_uniform=True,
        include_bias=False, K=40, H_xt=0.1, p=2, spatiotemporal_weights=variance,
    ), t


def _data(t):
    return np.column_stack([np.sin(2 * t) + 0.1 * t, np.cos(3 * t)])


def test_equal_components_reproduce_the_shared_whitener():
    _, t = _library(np.ones(401))
    x = _data(t)
    variance = 0.01 + 0.1 * np.abs(x[:, 0])
    shared, _ = _library(variance)
    per_eq, _ = _library(np.column_stack([variance, variance]))
    for lib in (shared, per_eq):
        lib.fit([x])
    theta_s, rhs_s = weak_system(shared, x)
    theta_p, rhs_p = weak_system(per_eq, x)
    assert per_eq.per_equation and not shared.per_equation
    assert theta_p.shape == (2,) + theta_s.shape
    for r in range(2):
        np.testing.assert_allclose(theta_p[r], theta_s, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(rhs_p, rhs_s, rtol=1e-10, atol=1e-12)


def test_each_equation_follows_its_own_variance():
    _, t = _library(np.ones(401))
    x = _data(t)
    var_a = 0.01 + 0.1 * np.abs(x[:, 0])
    var_b = 0.01 + 0.1 * np.abs(x[:, 1])
    per_eq, _ = _library(np.column_stack([var_a, var_b]))
    only_a, _ = _library(var_a)
    only_b, _ = _library(var_b)
    for lib in (per_eq, only_a, only_b):
        lib.fit([x])
    theta_p, rhs_p = weak_system(per_eq, x)
    theta_a, rhs_a = weak_system(only_a, x)
    theta_b, rhs_b = weak_system(only_b, x)
    np.testing.assert_allclose(theta_p[0], theta_a, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(theta_p[1], theta_b, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(rhs_p[:, 0], rhs_a[:, 0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(rhs_p[:, 1], rhs_b[:, 1], rtol=1e-10, atol=1e-12)
