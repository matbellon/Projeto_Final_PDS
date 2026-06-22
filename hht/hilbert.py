import numpy as np
from scipy.signal import hilbert as scipy_hilbert


def hilbert_transform(imfs, fs):
    """
    Apply the Hilbert transform to each IMF.

    Parameters
    ----------
    imfs : list of ndarray, each shape (N,)
    fs   : float — sampling frequency in Hz

    Returns
    -------
    amplitudes : ndarray, shape (n_imfs, N)
        Instantaneous amplitude (envelope) of each IMF.
    inst_freqs : ndarray, shape (n_imfs, N)
        Instantaneous frequency (Hz) of each IMF; negative values clipped to 0.
    """
    n_imfs = len(imfs)
    n = len(imfs[0])
    amplitudes = np.zeros((n_imfs, n))
    inst_freqs = np.zeros((n_imfs, n))

    for i, imf in enumerate(imfs):
        analytic = scipy_hilbert(imf)
        amplitudes[i] = np.abs(analytic)

        phase = np.unwrap(np.angle(analytic))
        dphase = np.diff(phase, prepend=phase[0])
        freq = dphase * fs / (2 * np.pi)
        freq = np.clip(freq, 0, None)
        inst_freqs[i] = freq

    return amplitudes, inst_freqs


def hilbert_spectrum(amplitudes, inst_freqs, fs, n_freqs=256):
    """
    Build the Hilbert spectrum H[freq_bin, time].

    Each sample's amplitude is deposited into the nearest frequency bin.

    Parameters
    ----------
    amplitudes  : ndarray, shape (n_imfs, N)
    inst_freqs  : ndarray, shape (n_imfs, N)
    fs          : float — sampling frequency in Hz
    n_freqs     : int — number of frequency bins (0 … fs/2)

    Returns
    -------
    H        : ndarray, shape (n_freqs, N)
    time_axis: ndarray, shape (N,) — seconds
    freq_axis: ndarray, shape (n_freqs,) — Hz
    """
    n_imfs, n = amplitudes.shape
    freq_axis = np.linspace(0, fs / 2, n_freqs)
    df = freq_axis[1] - freq_axis[0]

    H = np.zeros((n_freqs, n))

    for i in range(n_imfs):
        bins = np.round(inst_freqs[i] / df).astype(int)
        bins = np.clip(bins, 0, n_freqs - 1)
        for t in range(n):
            H[bins[t], t] += amplitudes[i, t]

    time_axis = np.arange(n) / fs
    return H, time_axis, freq_axis
