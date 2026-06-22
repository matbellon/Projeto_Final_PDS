import numpy as np
import pytest
from hht.detector import marginal_energy, detect_insertion

FS = 1000
T = np.linspace(0, 2, 2 * FS, endpoint=False)  # 2-second signal


def stationary_signal():
    """Pure 60 Hz sine — no nonlinear load."""
    return np.sin(2 * np.pi * 60 * T)


def signal_with_insertion(insertion_time=1.0):
    """
    60 Hz sine with a sudden harmonic burst starting at insertion_time.
    Simulates a non-linear load switching in.
    """
    sig = np.sin(2 * np.pi * 60 * T).copy()
    idx = int(insertion_time * FS)
    sig[idx:] += 0.6 * np.sin(2 * np.pi * 180 * T[idx:])
    sig[idx:] += 0.3 * np.sin(2 * np.pi * 300 * T[idx:])
    return sig


# ── Task 5: marginal_energy ───────────────────────────────────────────────────

def test_marginal_energy_shape():
    from hht.emd import emd
    from hht.hilbert import hilbert_transform
    sig = stationary_signal()
    imfs, _ = emd(sig, max_imfs=4)
    amps, _ = hilbert_transform(imfs, FS)
    window = FS // 60
    energy = marginal_energy(amps, window)
    assert energy.shape == amps.shape


def test_marginal_energy_constant_for_stationary():
    """Energy of a stationary sine should be nearly constant (CV < 10%)."""
    from hht.emd import emd
    from hht.hilbert import hilbert_transform
    sig = stationary_signal()
    imfs, _ = emd(sig, max_imfs=3)
    amps, _ = hilbert_transform(imfs, FS)
    window = FS // 60
    energy = marginal_energy(amps, window)
    total_energy = energy.sum(axis=0)
    center = slice(FS // 5, 4 * FS // 5)
    cv = total_energy[center].std() / (total_energy[center].mean() + 1e-12)
    assert cv < 0.10, f"CV = {cv:.3f} too high for stationary signal"


# ── Task 6: detect_insertion ──────────────────────────────────────────────────

def test_detect_insertion_no_event():
    """Stationary signal should return (None, None)."""
    sig = stationary_signal()
    t_star, idx = detect_insertion(sig, FS, threshold_factor=3.0)
    assert t_star is None
    assert idx is None


def test_detect_insertion_detects_event():
    """Should detect an event near the true insertion time."""
    true_time = 1.0
    sig = signal_with_insertion(true_time)
    t_star, idx = detect_insertion(sig, FS, threshold_factor=3.0)
    assert t_star is not None, "Insertion not detected"
    assert abs(t_star - true_time) < 0.1, \
        f"Detected at {t_star:.3f}s, expected ~{true_time}s"


def test_detect_insertion_returns_index():
    sig = signal_with_insertion(1.0)
    t_star, idx = detect_insertion(sig, FS)
    assert t_star is not None
    assert idx == int(round(t_star * FS))


def test_detect_insertion_any_length():
    """Works on signals with non-power-of-2 lengths (1537 samples)."""
    rng = np.random.default_rng(0)
    n = 1537
    t = np.linspace(0, n / FS, n, endpoint=False)
    sig = np.sin(2 * np.pi * 60 * t)
    insertion = n // 2
    sig[insertion:] += 0.8 * np.sin(2 * np.pi * 180 * t[insertion:])
    t_star, _ = detect_insertion(sig, FS, threshold_factor=2.5)
    assert t_star is not None
