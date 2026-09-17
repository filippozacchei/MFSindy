"""Binning rules for the bubble histogram.

The rungs differ by orders of magnitude, so one shared set of bins puts every
accurate rung in the first bin or two and shows nothing of their spread.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mfsindy.plots import bubble_hist


@pytest.fixture
def errors():
    rng = np.random.default_rng(0)
    return {
        "tight": 0.02 + 0.002 * rng.standard_normal(100),
        "wide": 1.5 + 0.4 * rng.standard_normal(100),
        "exact": np.zeros(100),
    }


def draw(errors, **kwargs):
    kwargs.setdefault("show", False)
    bubble_hist(errors_dict=errors, **kwargs)


def test_per_row_bins_resolve_a_tight_row(errors):
    """The tight row must not collapse into one bin of the shared range."""
    tight = errors["tight"]
    shared = np.linspace(0.0, float(errors["wide"].max()), 21)
    per_row = np.linspace(tight.min(), tight.max(), 21)
    occupied_shared = np.count_nonzero(np.histogram(tight, bins=shared)[0])
    occupied_per_row = np.count_nonzero(np.histogram(tight, bins=per_row)[0])
    assert occupied_shared <= 2
    assert occupied_per_row > occupied_shared


def test_draws_with_per_row_and_shared_bins(errors):
    draw(errors, n_bins=10)
    draw(errors, n_bins=10, shared_bins=True)


def test_a_row_of_one_repeated_value_is_drawn(errors):
    """Exact support recovery gives a row of zeros, an empty range to bin."""
    draw({"exact": errors["exact"], "wide": errors["wide"]}, n_bins=10)


def test_log_axis_rejects_non_positive_values(errors):
    with pytest.raises(ValueError, match="strictly positive"):
        draw(errors, n_bins=10, log_x=True)


def test_log_axis_accepts_positive_values(errors):
    draw({k: v for k, v in errors.items() if k != "exact"}, n_bins=10, log_x=True)
