"""
run_analysis.py
===============
Loads all project CSV signals, applies the HHT and the nonlinear-load
insertion detector, saves vector figures (PDF) for the LaTeX report, and
prints a summary table.

Usage
-----
    python run_analysis.py [--fs FS] [--threshold FACTOR] [--max-imfs N]

Defaults
--------
    --fs        : 3840  (64 samples per 60 Hz cycle — adjust if known)
    --threshold : 3.0
    --max-imfs  : 8
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hht import hht, detect_insertion


# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    return np.loadtxt(path)


def signal_time_axis(n, fs):
    return np.linspace(0, n / fs, n, endpoint=False)


def save_figure(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_signal_pair(sig_clean, sig_noisy, t, label, t_star, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, sig_clean, linewidth=0.8, color="steelblue")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{label} — sem ruído")
    axes[1].plot(t, sig_noisy, linewidth=0.8, color="tomato")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"{label} — com ruído")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        if t_star is not None:
            ax.axvline(t_star, color="black", linestyle="--",
                       linewidth=1.2, label=f"t* = {t_star*1000:.1f} ms")
            ax.legend(fontsize=8)
    axes[1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_hilbert_spectrum(H, time_axis, freq_axis, t_star, label, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = np.percentile(H, 99)
    im = ax.pcolormesh(time_axis, freq_axis, H,
                       shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Amplitude instantânea")
    if t_star is not None:
        ax.axvline(t_star, color="cyan", linestyle="--", linewidth=1.2,
                   label=f"t* = {t_star*1000:.1f} ms")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, min(freq_axis[-1], 600))
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    ax.set_title(f"Espectro de Hilbert — {label}")
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_marginal_energy(energy, t, t_star, label, out_path):
    combined = energy.sum(axis=0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, combined, linewidth=0.8, color="darkorange")
    if t_star is not None:
        ax.axvline(t_star, color="black", linestyle="--", linewidth=1.2,
                   label=f"t* = {t_star*1000:.1f} ms")
        ax.legend(fontsize=8)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Energia marginal")
    ax.set_title(f"Energia marginal das IMFs — {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HHT analysis of project signals")
    parser.add_argument("--fs", type=float, default=3840.0,
                        help="Sampling frequency in Hz (default: 3840)")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Detection threshold factor (default: 3.0)")
    parser.add_argument("--max-imfs", type=int, default=8,
                        help="Maximum number of IMFs (default: 8)")
    args = parser.parse_args()

    fs = args.fs
    threshold_factor = args.threshold
    max_imfs = args.max_imfs

    base_dir = os.path.dirname(os.path.abspath(__file__))
    signal_dir = os.path.join(base_dir, "sinais", "sinais")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HHT Analysis — Grupo A9")
    print(f"  fs = {fs} Hz | threshold = {threshold_factor}*sigma | max_imfs = {max_imfs}")
    print(f"{'='*60}\n")

    header = f"{'Sinal':<25} {'t* (ms)':>10} {'Amostra':>9}"
    print(header)
    print("-" * len(header))

    for i in range(1, 10):
        clean_path = os.path.join(signal_dir, f"sinal_{i}_semruido.csv")
        noisy_path = os.path.join(signal_dir, f"sinal_{i}_ruido.csv")

        if not os.path.exists(clean_path):
            continue

        label = f"sinal_{i}"
        print(f"\nProcessando {label}...")

        sig_clean = load_csv(clean_path)
        sig_noisy = load_csv(noisy_path) if os.path.exists(noisy_path) else sig_clean
        n = len(sig_clean)
        t = signal_time_axis(n, fs)

        # Run HHT on noisy signal (more realistic) for detection
        (imfs, residue, amplitudes, inst_freqs,
         H, time_axis, freq_axis) = hht(sig_noisy, fs,
                                         max_imfs=max_imfs)

        t_star, idx = detect_insertion(sig_noisy, fs,
                                       threshold_factor=threshold_factor,
                                       max_imfs=max_imfs)

        t_star_ms = f"{t_star*1000:.2f}" if t_star is not None else "—"
        idx_str = str(idx) if idx is not None else "—"
        print(f"  {label + ' (ruído)':<23} {t_star_ms:>10} {idx_str:>9}")

        # figures
        plot_signal_pair(
            sig_clean, sig_noisy, t, label, t_star,
            os.path.join(fig_dir, f"{label}_sinais.pdf")
        )
        plot_hilbert_spectrum(
            H, time_axis, freq_axis, t_star, label,
            os.path.join(fig_dir, f"{label}_espectro_hilbert.pdf")
        )

        from hht.detector import marginal_energy
        energy = marginal_energy(amplitudes, max(1, int(fs // 60)))
        plot_marginal_energy(
            energy, t, t_star, label,
            os.path.join(fig_dir, f"{label}_energia_marginal.pdf")
        )

    print(f"\n{'='*60}")
    print(f"  Figuras salvas em: {fig_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
