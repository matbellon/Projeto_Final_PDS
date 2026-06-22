import numpy as np
from scipy.interpolate import CubicSpline


def _find_extrema(signal):
    """Return indices of local maxima and minima."""
    n = len(signal)
    maxima, minima = [], []
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            maxima.append(i)
        elif signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            minima.append(i)
    return np.array(maxima, dtype=int), np.array(minima, dtype=int)


def _mirror_pad(indices, values, n):
    """Extend extrema arrays with mirrored endpoints to reduce edge effects."""
    if len(indices) < 2:
        return indices, values

    left_idx = 2 * indices[0] - indices[1]
    right_idx = 2 * indices[-1] - indices[-2]
    left_idx = max(0, left_idx)
    right_idx = min(n - 1, right_idx)

    indices = np.concatenate([[left_idx], indices, [right_idx]])
    values = np.concatenate([[values[0]], values, [values[-1]]])
    return indices, values


def _envelope(signal, indices, values):
    """Cubic spline envelope through the given extrema."""
    t = np.arange(len(signal))
    if len(indices) < 4:
        return np.full(len(signal), np.mean(values))
    cs = CubicSpline(indices, values, bc_type='not-a-knot')
    return cs(t)


def _is_imf(signal, upper_env, lower_env, tol=0.2):
    """Check IMF stopping criterion: mean envelope ≈ 0."""
    mean_env = (upper_env + lower_env) / 2
    sd = np.sum(mean_env ** 2) / (np.sum(signal ** 2) + 1e-12)
    return sd < tol


def _is_monotonic(signal):
    diff = np.diff(signal)
    return np.all(diff >= 0) or np.all(diff <= 0)


def _sift(signal, sd_threshold=0.2, max_iter=100):
    """
    Extract one IMF from signal via the sifting process.

    Returns the IMF as an array of the same length as signal.
    """
    h = signal.copy()
    n = len(h)
    t = np.arange(n)

    for _ in range(max_iter):
        max_idx, min_idx = _find_extrema(h)

        if len(max_idx) < 2 or len(min_idx) < 2:
            break

        max_idx, max_val = _mirror_pad(max_idx, h[max_idx], n)
        min_idx, min_val = _mirror_pad(min_idx, h[min_idx], n)

        upper = _envelope(h, max_idx, max_val)
        lower = _envelope(h, min_idx, min_val)

        if _is_imf(h, upper, lower, sd_threshold):
            break

        mean_env = (upper + lower) / 2
        h = h - mean_env

    return h


def emd(signal, max_imfs=None, sd_threshold=0.2):
    """
    Empirical Mode Decomposition.

    Decomposes `signal` into a list of IMFs and a residue such that
    sum(imfs) + residue == signal (up to floating-point precision).

    Parameters
    ----------
    signal : array-like, shape (N,)
    max_imfs : int or None
        Maximum number of IMFs to extract. None means extract all.
    sd_threshold : float
        Stopping criterion for the sifting process.

    Returns
    -------
    imfs : list of ndarray, each shape (N,)
    residue : ndarray, shape (N,)
    """
    signal = np.asarray(signal, dtype=float)
    residue = signal.copy()
    imfs = []

    while True:
        if max_imfs is not None and len(imfs) >= max_imfs:
            break
        if _is_monotonic(residue):
            break

        max_idx, min_idx = _find_extrema(residue)
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

        imf = _sift(residue, sd_threshold=sd_threshold)
        imfs.append(imf)
        residue = residue - imf

    return imfs, residue
