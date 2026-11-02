"""Pointwise noise variance estimated from the data, for when it is not known.

The weighted rungs whiten by the noise variance of every sample. When that is not
given, it is estimated from each observed series on its own: the residual from a
local smoother is mostly noise, and its local mean square, corrected for what the
smoother itself removes from white noise, estimates the variance.

- Smoother: Savitzky-Golay, cubic over ``window`` samples. A cubic follows the
  signal's curvature, so the residual keeps little of the signal as long as the
  window is short next to its time scale.
- Variance: the moving average of the squared residual over ``variance_window``
  samples, divided by ``1 - 2 h_0 + sum_k h_k^2`` (h the smoother's weights): the
  expected square of the residual of white noise of unit variance, since the
  residual is ``(I - S) e``.

The estimate is fixed by these three numbers, not tuned to any case.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_coeffs, savgol_filter


def estimate_noise_variance(x: np.ndarray, window: int = 11, polyorder: int = 3,
                            variance_window: int = 21, floor: float = 1e-12) -> np.ndarray:
    """Pointwise noise variance of the series ``x`` (n_samples, n_components),
    estimated from the series alone; same shape as ``x``."""

    x = np.asarray(x, dtype=float)
    residual = x - savgol_filter(x, window, polyorder, axis=0, mode="interp")
    h = savgol_coeffs(window, polyorder)
    shrink = 1.0 - 2.0 * h[window // 2] + float(np.sum(h ** 2))
    variance = uniform_filter1d(residual ** 2, variance_window, axis=0, mode="nearest") / shrink
    return np.maximum(variance, floor)


__all__ = ["estimate_noise_variance"]
