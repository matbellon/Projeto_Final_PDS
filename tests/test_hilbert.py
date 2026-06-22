import numpy as np
import pytest
from hht.hilbert import hilbert_transform, hilbert_spectrum

FS = 1000
T = np.linspace(0, 1, FS, endpoint=False)


def make_imfs():
    imf1 = np.sin(2 * np.pi * 60 * T)
    imf2 = 0.5 * np.sin(2 * np.pi * 180 * T)
    return [imf1, imf2]


# ── Task 3 ───────────────────────────────────────────────────────────────────

def test_hilbert_transform_shapes():
    imfs = make_imfs()
    amps, freqs = hilbert_transform(imfs, FS)
    assert amps.shape == (2, FS)
    assert freqs.shape == (2, FS)


def test_hilbert_transform_amplitude_sine():
    """Amplitude of a unit sine should be ≈ 1 (excluding edges)."""
    imf = [np.sin(2 * np.pi * 60 * T)]
    amps, _ = hilbert_transform(imf, FS)
    center = slice(FS // 10, 9 * FS // 10)
    assert np.allclose(amps[0, center], 1.0, atol=0.05)


def test_hilbert_transform_freq_sine():
    """Instantaneous frequency of a 60 Hz sine should be ≈ 60 Hz."""
    imf = [np.sin(2 * np.pi * 60 * T)]
    _, freqs = hilbert_transform(imf, FS)
    center = slice(FS // 10, 9 * FS // 10)
    assert np.allclose(freqs[0, center], 60.0, atol=2.0)


def test_hilbert_transform_no_negative_freqs():
    imfs = make_imfs()
    _, freqs = hilbert_transform(imfs, FS)
    assert np.all(freqs >= 0)


# ── Task 4 ───────────────────────────────────────────────────────────────────

def test_hilbert_spectrum_shape():
    imfs = make_imfs()
    amps, freqs = hilbert_transform(imfs, FS)
    H, t_ax, f_ax = hilbert_spectrum(amps, freqs, FS, n_freqs=128)
    assert H.shape == (128, FS)
    assert len(t_ax) == FS
    assert len(f_ax) == 128


def test_hilbert_spectrum_energy_at_60hz():
    """Energy should be concentrated near the 60 Hz bin."""
    imf = [np.sin(2 * np.pi * 60 * T)]
    amps, freqs = hilbert_transform(imf, FS)
    H, _, f_ax = hilbert_spectrum(amps, freqs, FS, n_freqs=256)
    energy_per_freq = H.sum(axis=1)
    peak_freq = f_ax[np.argmax(energy_per_freq)]
    assert abs(peak_freq - 60) < 5.0
