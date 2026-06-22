import numpy as np
from .hht import hht


def marginal_energy(amplitudes, window_size):
    """
    Smoothed instantaneous energy per IMF via a moving-average of A²(t).

    Parameters
    ----------
    amplitudes  : ndarray, shape (n_imfs, N)
    window_size : int — number of samples for the moving average

    Returns
    -------
    energy : ndarray, shape (n_imfs, N)
    """
    kernel = np.ones(window_size) / window_size
    energy = np.zeros_like(amplitudes)
    for i in range(amplitudes.shape[0]):
        sq = amplitudes[i] ** 2
        energy[i] = np.convolve(sq, kernel, mode='same')
    return energy


def detect_insertion(signal, fs, threshold_factor=3.0, window_size=None,
                     n_high_imfs=3, max_imfs=8):
    """
    Detect the time instant at which a non-linear load was switched in.

    Algorithm
    ---------
    1. Run the full HHT on the signal.
    2. Compute marginal energy for each IMF.
    3. Sum energy of the first `n_high_imfs` IMFs (high-frequency content).
    4. Estimate baseline statistics from the first 20% of the signal.
    5. Return the first sample where energy > μ + threshold_factor·σ.

    Parameters
    ----------
    signal           : array-like, shape (N,)
    fs               : float — sampling frequency in Hz
    threshold_factor : float — sensitivity (lower → more sensitive)
    window_size      : int or None — moving-average window (default fs//60)
    n_high_imfs      : int — number of high-frequency IMFs to aggregate
    max_imfs         : int — max IMFs extracted by EMD

    Returns
    -------
    t_seconds    : float or None — insertion time in seconds
    sample_index : int or None — insertion sample index
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if window_size is None:
        window_size = max(1, int(fs // 60))

    imfs, residue, amplitudes, inst_freqs, H, time_axis, freq_axis = hht(
        signal, fs, max_imfs=max_imfs
    )

    if len(imfs) == 0:
        return None, None

    energy = marginal_energy(amplitudes, window_size)

    n_use = min(n_high_imfs, len(imfs))
    combined = energy[:n_use].sum(axis=0)

    baseline_end = max(1, int(0.20 * n))
    mu = combined[:baseline_end].mean()
    sigma = combined[:baseline_end].std()

    threshold = mu + threshold_factor * sigma

    candidates = np.where(combined[baseline_end:] > threshold)[0]
    if len(candidates) == 0:
        return None, None

    sample_index = baseline_end + int(candidates[0])
    t_seconds = sample_index / fs
    return t_seconds, sample_index
