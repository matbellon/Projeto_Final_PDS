import numpy as np
from .hht import hht


def marginal_energy(amplitudes, window_size):
    """Energia instantânea suavizada de cada IMF.

    Faz uma média móvel de A²(t) com janela de window_size amostras, devolvendo
    um perfil de energia por IMF ao longo do tempo.
    """
    kernel = np.ones(window_size) / window_size
    energy = np.zeros_like(amplitudes)
    for i in range(amplitudes.shape[0]):
        sq = amplitudes[i] ** 2
        energy[i] = np.convolve(sq, kernel, mode='same')
    return energy


def detect_insertion(signal, fs, threshold_factor=3.0, window_size=None,
                     n_high_imfs=3, max_imfs=8):
    """Detecta o instante em que uma carga não linear entrou no circuito.

    A ideia: rodar a HHT, somar a energia marginal das IMFs de alta frequência
    (as primeiras n_high_imfs, que carregam os harmônicos e transientes) e olhar
    onde essa energia dispara. A linha de base (média mu e desvio sigma) é
    estimada nos primeiros 20% do sinal, assumidos como trecho estacionário.
    O instante detectado é a primeira amostra onde a energia passa de
    mu + threshold_factor*sigma.

    Retorna (t em segundos, índice da amostra) ou (None, None) se não achar nada.
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
