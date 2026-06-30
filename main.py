"""
Trabalho de Processamento Digital de Sinais - SEL0615
Análise HHT do arquivo sinal_9_semruido.csv do Grupo A9.

Integrantes do Grupo:
- Mateus Bellon Liparizi    - 15478811
- Luisa Domingues Santello  - 15472818
- Giovanna Bragatto Piva    - 14108847
- João Vitor Bartsch Morasi - 

Resumo do Funcionamento:
- Roda a HHT e o detector de inserção de carga não linear sobre o sinal.
- Salva as figuras em PDF para o relatório
- Imprime no terminal o instante detectado para inserção da carga.

Chamada Reduzida: 
- python main.py [--fs FS] [--threshold FATOR] [--max-imfs N]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sem janela: só salvamos os arquivos
import matplotlib.pyplot as plt

from hht import hht, detect_insertion
from hht.detector import marginal_energy


def load_csv(path):
    return np.loadtxt(path)


def signal_time_axis(n, fs):
    return np.linspace(0, n / fs, n, endpoint=False)


def mark_instant(ax, t_star, color="black"):
    # Marca o instante de inserção t* com uma linha vertical tracejada.
    if t_star is not None:
        ax.axvline(t_star, color=color, linestyle="--", linewidth=1.2,
                   label=f"t* = {t_star*1000:.1f} ms")
        ax.legend(fontsize=8)


def save_figure(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  figura salva: {path}")


def plot_signal(sig, t, label, t_star, out_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, sig, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{label} — sem ruído")
    ax.grid(True, alpha=0.3)
    mark_instant(ax, t_star)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_hilbert_spectrum(H, time_axis, freq_axis, t_star, label, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    # Satura o colormap no percentil 99 para o transiente não ofuscar o resto.
    im = ax.pcolormesh(time_axis, freq_axis, H, shading="auto",
                       cmap="inferno", vmin=0, vmax=np.percentile(H, 99))
    fig.colorbar(im, ax=ax, label="Amplitude instantânea")
    mark_instant(ax, t_star, color="cyan")
    ax.set_ylim(0, min(freq_axis[-1], 600))
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    ax.set_title(f"Espectro de Hilbert — {label}")
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_marginal_energy(energy, t, t_star, label, out_path):
    # Soma a energia de todas as IMFs para ter um único perfil no tempo.
    combined = energy.sum(axis=0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, combined, linewidth=0.8, color="darkorange")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Energia marginal")
    ax.set_title(f"Energia marginal das IMFs — {label}")
    ax.grid(True, alpha=0.3)
    mark_instant(ax, t_star)
    fig.tight_layout()
    save_figure(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description="Análise HHT do sinal do Grupo A9")
    parser.add_argument("--fs", type=float, default=7680.0,
                        help="Taxa de amostragem em Hz")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Fator de limiar da detecção (mu + fator*sigma)")
    parser.add_argument("--max-imfs", type=int, default=8,
                        help="Número máximo de IMFs")
    args = parser.parse_args()

    fs = args.fs
    threshold_factor = args.threshold
    max_imfs = args.max_imfs

    base_dir = os.path.dirname(os.path.abspath(__file__))
    signal_dir = os.path.join(base_dir, "signal")
    fig_dir = os.path.join(base_dir, "plots")
    os.makedirs(fig_dir, exist_ok=True)

    # O Grupo A9 trabalha só com o sinal 9 sem ruído.
    label = "sinal_9"
    clean_path = os.path.join(signal_dir, f"{label}_semruido.csv")

    print(f"\nfs = {fs} Hz | limiar = mu + {threshold_factor}*sigma | max_imfs = {max_imfs}\n")
    print(f"Processando {label}...")

    sig = load_csv(clean_path)
    n = len(sig)
    t = signal_time_axis(n, fs)

    (imfs, residue, amplitudes, inst_freqs,
     H, time_axis, freq_axis) = hht(sig, fs, max_imfs=max_imfs)

    t_star, idx = detect_insertion(sig, fs,
                                   threshold_factor=threshold_factor,
                                   max_imfs=max_imfs)

    if t_star is not None:
        print(f"  inserção detectada em t* = {t_star*1000:.2f} ms (amostra {idx})")
    else:
        print("  nenhuma inserção detectada")

    plot_signal(
        sig, t, label, t_star,
        os.path.join(fig_dir, f"{label}_sinal.pdf")
    )
    plot_hilbert_spectrum(
        H, time_axis, freq_axis, t_star, label,
        os.path.join(fig_dir, f"{label}_espectro_hilbert.pdf")
    )

    # Janela de 1 ciclo da fundamental (60 Hz) para suavizar a energia.
    energy = marginal_energy(amplitudes, max(1, int(fs // 60)))
    plot_marginal_energy(
        energy, t, t_star, label,
        os.path.join(fig_dir, f"{label}_energia_marginal.pdf")
    )

    print(f"\nFiguras salvas em: {fig_dir}\n")


if __name__ == "__main__":
    main()
