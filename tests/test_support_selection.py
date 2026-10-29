"""The support rule skips supports whose weak system is too short to identify the model."""
from types import SimpleNamespace

import numpy as np

from mfsindy.experiments.selection import BICEnsembleSTLSQ, select_support


def _stub(n_rows, spread, n_terms=9):
    return SimpleNamespace(n_rows_=n_rows, n_terms_=n_terms, spread_=spread)


def test_underdetermined_support_is_not_chosen_even_with_the_smallest_spread():
    fits = {0.05: {"HF": _stub(20, 0.3)}, 0.2: {"HF": _stub(5, 0.01)}}
    assert select_support(fits) == {"HF": 0.05}


def test_with_no_admissible_support_every_support_is_a_candidate():
    fits = {0.1: {"HF": _stub(10, 0.3)}, 0.2: {"HF": _stub(5, 0.1)}}
    assert select_support(fits) == {"HF": 0.2}


def test_the_optimizer_records_its_system_size():
    np.random.seed(0)
    G = np.random.standard_normal((40, 4))
    y = G @ np.array([1.0, -2.0, 0.0, 0.5])
    opt = BICEnsembleSTLSQ(n_models=5, n_thresholds=5).fit(G, y[:, None])
    assert (opt.n_rows_, opt.n_terms_) == (40, 4)
