from .emd import emd
from .hilbert import hilbert_transform, hilbert_spectrum


def hht(signal, fs, max_imfs=None, sd_threshold=0.2, n_freqs=256):
    """
    Hilbert-Huang Transform.

    Applies EMD followed by the Hilbert transform and assembles the
    Hilbert spectrum. Works on signals of any length.

    Parameters
    ----------
    signal       : array-like, shape (N,)
    fs           : float — sampling frequency in Hz
    max_imfs     : int or None — maximum number of IMFs (None = all)
    sd_threshold : float — sifting stopping criterion
    n_freqs      : int — frequency resolution of the Hilbert spectrum

    Returns
    -------
    imfs       : list of ndarray, each shape (N,)
    residue    : ndarray, shape (N,)
    amplitudes : ndarray, shape (n_imfs, N) — instantaneous amplitude
    inst_freqs : ndarray, shape (n_imfs, N) — instantaneous frequency (Hz)
    H          : ndarray, shape (n_freqs, N) — Hilbert spectrum
    time_axis  : ndarray, shape (N,) — seconds
    freq_axis  : ndarray, shape (n_freqs,) — Hz
    """
    imfs, residue = emd(signal, max_imfs=max_imfs, sd_threshold=sd_threshold)
    amplitudes, inst_freqs = hilbert_transform(imfs, fs)
    H, time_axis, freq_axis = hilbert_spectrum(
        amplitudes, inst_freqs, fs, n_freqs=n_freqs
    )
    return imfs, residue, amplitudes, inst_freqs, H, time_axis, freq_axis
