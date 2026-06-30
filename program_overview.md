# Transformada de Hilbert-Huang — Detecção de Cargas Não Lineares (Grupo A9)

Integrantes do Grupo:
- Mateus Bellon Liparizi    - 15478811
- Luisa Domingues Santello  - 15472818
- Giovanna Bragatto Piva    - 14108847
- João Vitor Bartsch Morasi - 

## Objectives

O programa detecta automaticamente o instante em que uma carga não linear é
inserida em um sinal elétrico de 60 Hz, usando a **Transformada de Hilbert-Huang
(HHT)** implementada do zero.

A HHT trabalha em dois passos. Primeiro a **EMD (Empirical Mode Decomposition)**
quebra o sinal em componentes oscilantes chamadas **IMFs (Intrinsic Mode
Functions)**, que se adaptam aos dados (ao contrário da FFT, que usa base fixa).
Depois a **Transformada de Hilbert** é aplicada a cada IMF para obter amplitude e
frequência instantâneas, montando uma representação tempo-frequência (espectro de
Hilbert). Como cargas não lineares introduzem harmônicos e transientes, elas
aparecem como um salto de energia nas IMFs de alta frequência — é esse salto que
o detector procura.

## Workflow

Fluxo: carrega o CSV → roda a HHT → mede a energia das IMFs rápidas → detecta o
instante `t*` onde a energia dispara → gera as figuras.

Divisão dos arquivos:

| Arquivo | Responsabilidade |
|---------|------------------|
| `hht/emd.py` | EMD: sifting (`_sift`) e decomposição iterativa (`emd`) |
| `hht/hilbert.py` | Transformada de Hilbert por IMF e montagem do espectro |
| `hht/hht.py` | Função `hht()` que junta EMD + Hilbert + espectro |
| `hht/detector.py` | Energia marginal e detecção do instante de inserção |
| `main.py` | Carrega `sinal_9_semruido.csv`, roda a pipeline e salva os PDFs em `plots/` |

Uso:
```
python main.py [--fs FS] [--threshold FATOR] [--max-imfs N]
```

## Equations

**Reconstrução pela EMD** — o sinal é a soma das IMFs mais o resíduo:
```
x(t) = sum_{k=1}^{K} IMF_k(t) + r(t)
```

**Critério de parada do sifting** — razão entre a média dos envelopes e o sinal:
```
SD = sum_t [ m(t)^2 ] / sum_t [ h(t)^2 ]  <  sd_threshold   (padrão 0.2)
```

**Sinal analítico** (via transformada de Hilbert) e grandezas instantâneas:
```
z_k(t) = c_k(t) + j * H{c_k(t)}
A_k(t) = |z_k(t)|                         (amplitude)
f_k(t) = (1 / 2*pi*fs) * d/dt unwrap(phi_k)   (frequência, valores < 0 zerados)
```

**Energia marginal** — média móvel de A²(t) com janela de 1 ciclo (`fs/60`):
```
E_k(t) = (1/W) * sum_{tau} A_k(tau)^2
```

**Detecção do instante** — limiar estatístico sobre os primeiros 20% (base):
```
mu, sigma = media e desvio de E_total nos primeiros 20% do sinal
t* = primeiro instante onde E_total(t) > mu + threshold_factor * sigma
```
