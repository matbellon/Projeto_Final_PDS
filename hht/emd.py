import numpy as np
from scipy.interpolate import CubicSpline


def _find_extrema(signal):
    """Acha os índices dos máximos e mínimos locais do sinal."""
    n = len(signal)
    maxima, minima = [], []
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            maxima.append(i)
        elif signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            minima.append(i)
    return np.array(maxima, dtype=int), np.array(minima, dtype=int)


def _mirror_pad(indices, values, n):
    """Espelha um extremo em cada ponta para diminuir o efeito de borda."""
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
    """Envelope por spline cúbica passando pelos extremos dados."""
    t = np.arange(len(signal))
    if len(indices) < 4:
        return np.full(len(signal), np.mean(values))
    cs = CubicSpline(indices, values, bc_type='not-a-knot')
    return cs(t)


def _is_imf(signal, upper_env, lower_env, tol=0.2):
    """Critério de parada do sifting: média dos envelopes perto de zero."""
    mean_env = (upper_env + lower_env) / 2
    sd = np.sum(mean_env ** 2) / (np.sum(signal ** 2) + 1e-12)
    return sd < tol


def _is_monotonic(signal):
    diff = np.diff(signal)
    return np.all(diff >= 0) or np.all(diff <= 0)


def _sift(signal, sd_threshold=0.2, max_iter=100):
    """Extrai uma IMF do sinal pelo processo de peneiramento (sifting).

    Retorna a IMF com o mesmo tamanho do sinal de entrada.
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
    """Decomposição Empírica de Modos (EMD).

    Quebra o sinal numa lista de IMFs mais um resíduo, de forma que a soma das
    IMFs com o resíduo reconstrói o sinal original (a menos de erro numérico).
    Com max_imfs=None, extrai todas as IMFs possíveis; sd_threshold é o critério
    de parada do sifting.
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
