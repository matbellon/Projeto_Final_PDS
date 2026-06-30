import numpy as np
from scipy.signal import hilbert as scipy_hilbert


def hilbert_transform(imfs, fs):
    """Aplica a transformada de Hilbert em cada IMF.

    Devolve a amplitude instantânea (envelope) e a frequência instantânea em Hz
    de cada IMF. Frequências negativas, que aparecem como artefato nas bordas,
    são zeradas.
    """
    n_imfs = len(imfs)
    n = len(imfs[0])
    amplitudes = np.zeros((n_imfs, n))
    inst_freqs = np.zeros((n_imfs, n))

    for i, imf in enumerate(imfs):
        analytic = scipy_hilbert(imf)
        amplitudes[i] = np.abs(analytic)

        # Frequência instantânea = derivada da fase do sinal analítico.
        phase = np.unwrap(np.angle(analytic))
        dphase = np.diff(phase, prepend=phase[0])
        freq = dphase * fs / (2 * np.pi)
        inst_freqs[i] = np.clip(freq, 0, None)

    return amplitudes, inst_freqs


def hilbert_spectrum(amplitudes, inst_freqs, fs, n_freqs=256):
    """Monta o espectro de Hilbert H[frequência, tempo].

    Para cada instante, a amplitude de cada IMF é jogada na faixa de frequência
    mais próxima. O eixo de frequência vai de 0 a fs/2 dividido em n_freqs faixas.
    Devolve a matriz H e os eixos de tempo (s) e frequência (Hz).
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
