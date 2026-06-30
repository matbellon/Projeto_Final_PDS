from .emd import emd
from .hilbert import hilbert_transform, hilbert_spectrum


def hht(signal, fs, max_imfs=None, sd_threshold=0.2, n_freqs=256):
    """Transformada de Hilbert-Huang.

    Junta os três passos: EMD para tirar as IMFs, transformada de Hilbert para
    pegar amplitude e frequência instantâneas, e a montagem do espectro de
    Hilbert. Funciona com sinais de qualquer tamanho.

    Devolve, nesta ordem: imfs, resíduo, amplitudes, frequências instantâneas,
    espectro H e os eixos de tempo e frequência.
    """
    imfs, residue = emd(signal, max_imfs=max_imfs, sd_threshold=sd_threshold)
    amplitudes, inst_freqs = hilbert_transform(imfs, fs)
    H, time_axis, freq_axis = hilbert_spectrum(
        amplitudes, inst_freqs, fs, n_freqs=n_freqs
    )
    return imfs, residue, amplitudes, inst_freqs, H, time_axis, freq_axis
