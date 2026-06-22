import numpy as np
import pytest
from hht.emd import _sift, emd


FS = 1000
T = np.linspace(0, 1, FS, endpoint=False)


def sine(freq, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * T)


# ── Task 1: _sift ────────────────────────────────────────────────────────────

def test_sift_returns_same_length():
    sig = sine(60)
    imf = _sift(sig)
    assert imf.shape == sig.shape


def test_sift_pure_sine_recovers_signal():
    """For a pure sinusoid the first sift should be close to the original."""
    sig = sine(60)
    imf = _sift(sig)
    # Allow up to 5% RMS error
    rms_error = np.sqrt(np.mean((imf - sig) ** 2))
    rms_signal = np.sqrt(np.mean(sig ** 2))
    assert rms_error / rms_signal < 0.05


def test_sift_sd_threshold_parameter():
    """Stricter threshold should still return an array of the right shape."""
    sig = sine(60) + 0.3 * sine(180)
    imf = _sift(sig, sd_threshold=0.05)
    assert imf.shape == sig.shape


# ── Task 2: emd ──────────────────────────────────────────────────────────────

def test_emd_reconstruction():
    """sum(IMFs) + residue must equal the original signal."""
    sig = sine(60) + 0.5 * sine(180) + 0.2 * sine(300)
    imfs, residue = emd(sig)
    reconstructed = sum(imfs) + residue
    assert np.allclose(reconstructed, sig, atol=1e-10), \
        f"Max reconstruction error: {np.max(np.abs(reconstructed - sig))}"


def test_emd_reconstruction_arbitrary_length():
    """Works for non-power-of-2 signal lengths."""
    rng = np.random.default_rng(42)
    sig = np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1537, endpoint=False))
    sig += 0.3 * rng.standard_normal(1537)
    imfs, residue = emd(sig, max_imfs=5)
    reconstructed = sum(imfs) + residue
    assert np.allclose(reconstructed, sig, atol=1e-10)


def test_emd_max_imfs():
    sig = sine(60) + 0.5 * sine(180) + 0.2 * sine(300)
    imfs, _ = emd(sig, max_imfs=2)
    assert len(imfs) <= 2


def test_emd_produces_at_least_one_imf():
    sig = sine(60) + 0.5 * sine(180)
    imfs, residue = emd(sig)
    assert len(imfs) >= 1


def test_emd_csv_signal():
    """Reconstruction holds on the actual project CSV."""
    import os
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "..", "sinais", "sinais", "sinal_1_semruido.csv"
    )
    sig = np.loadtxt(csv_path)
    imfs, residue = emd(sig, max_imfs=8)
    reconstructed = sum(imfs) + residue
    assert np.allclose(reconstructed, sig, atol=1e-10), \
        f"CSV reconstruction error: {np.max(np.abs(reconstructed - sig))}"
