# Implementation Plan: Hilbert-Huang Transform — Detecção de Cargas Não Lineares (Grupo A9)

## Overview

Implementar a Transformada de Hilbert-Huang (HHT) em Python para extrair componentes no domínio da frequência de sinais elétricos e, com base na análise tempo-frequência resultante, propor um algoritmo capaz de identificar o instante de inserção de cargas não lineares.

A HHT é composta por dois estágios:
1. **Empirical Mode Decomposition (EMD):** decompõe o sinal em Intrinsic Mode Functions (IMFs), adaptativas e não estacionárias.
2. **Transformada de Hilbert (HT):** aplicada a cada IMF para obter frequência instantânea e amplitude instantânea → espectro de Hilbert (representação tempo-frequência).

A detecção do instante de inserção baseia-se no fato de que cargas não lineares introduzem distorção harmônica e transientes bruscos, que se manifestam como variações abruptas de energia e frequência instantânea nas IMFs.

---

## Architecture Decisions

- **Linguagem:** Python 3.x com NumPy, SciPy e Matplotlib.
- **Sem bibliotecas EMD de terceiros:** a EMD será implementada do zero para cumprir o requisito do enunciado.
- **Interface genérica:** todas as funções aceitam arrays 1-D de qualquer comprimento e a taxa de amostragem `fs` como parâmetro.
- **Separação de responsabilidades:** EMD, HT e detector de inserção são funções independentes e testáveis individualmente.

---

## Task List

### Phase 1: EMD — Empirical Mode Decomposition

---

#### Task 1: Implementar o algoritmo de sifting (extração de uma IMF)

**Description:**  
Implementar a função `_sift(signal)` que executa o processo de peneiramento (sifting) para extrair uma única IMF a partir de um sinal. O sifting inclui: detecção de extremos locais, interpolação cúbica dos envelopes superior e inferior, subtração da média dos envelopes e critério de parada (desvio padrão entre iterações consecutivas).

**Acceptance criteria:**
- [ ] A função retorna um array com o mesmo comprimento do sinal de entrada.
- [ ] O resultado satisfaz as duas condições de IMF: número de zeros e extremos diferem por no máximo 1; média dos envelopes ≈ 0.
- [ ] Critério de parada configurável via parâmetro `sd_threshold` (padrão 0.2).

**Verification:**
- [ ] Teste com sinal sintético senoidal de 60 Hz: a primeira IMF deve recuperar a senoide original.
- [ ] Teste com sinal de dois tons (60 Hz + harmônico): as IMFs devem separar as componentes.

**Dependencies:** None

**Files likely touched:**
- `hht/emd.py`
- `tests/test_emd.py`

**Estimated scope:** Medium (3-4 files)

---

#### Task 2: Implementar EMD completa — decomposição iterativa

**Description:**  
Implementar a função `emd(signal, max_imfs=None, sd_threshold=0.2)` que aplica `_sift` iterativamente ao resíduo do sinal até que o resíduo seja monotônico ou o número máximo de IMFs seja atingido. Retorna lista de IMFs e o resíduo final.

**Acceptance criteria:**
- [ ] `sum(IMFs) + resíduo == sinal original` (erro numérico < 1e-10).
- [ ] Funciona para sinais de qualquer comprimento (potência de 2 ou não).
- [ ] `max_imfs=None` extrai todas as IMFs possíveis.

**Verification:**
- [ ] Verificar reconstrução: `np.allclose(sum(imfs) + residue, signal, atol=1e-10)`.
- [ ] Teste com os CSVs do projeto (`sinal_1_semruido.csv`) e checar número de IMFs geradas.

**Dependencies:** Task 1

**Files likely touched:**
- `hht/emd.py`
- `tests/test_emd.py`

**Estimated scope:** Small (2 files)

---

### Checkpoint: Phase 1
- [ ] `tests/test_emd.py` passa sem erros.
- [ ] Reconstrução exata verificada para ao menos 3 sinais do projeto.
- [ ] Revisão humana antes de prosseguir.

---

### Phase 2: Transformada de Hilbert e Espectro de Hilbert

---

#### Task 3: Aplicar a Transformada de Hilbert a cada IMF

**Description:**  
Implementar `hilbert_transform(imfs)` que recebe a lista de IMFs e aplica `scipy.signal.hilbert` a cada uma, retornando amplitude instantânea `A(t)` e frequência instantânea `f(t)` para cada IMF.

**Acceptance criteria:**
- [ ] Retorna dois arrays `(amplitudes, freqs)`, cada um de shape `(n_imfs, n_samples)`.
- [ ] Frequência instantânea calculada como derivada da fase analítica, normalizada por `2π·fs`.
- [ ] Frequências negativas zeradas (artefato de bordas).

**Verification:**
- [ ] Teste com senoide de 60 Hz a fs=1000 Hz: frequência instantânea deve ser ≈ 60 Hz em toda a extensão central do sinal.

**Dependencies:** Task 2

**Files likely touched:**
- `hht/hilbert.py`
- `tests/test_hilbert.py`

**Estimated scope:** Small (2 files)

---

#### Task 4: Montar o Espectro de Hilbert (representação tempo-frequência)

**Description:**  
Implementar `hilbert_spectrum(amplitudes, freqs, fs, n_freqs=256)` que gera uma matriz tempo-frequência distribuindo a amplitude instantânea de cada IMF na frequência correspondente ao longo do tempo. Saída: matriz `H[f, t]` e vetores de eixos `time_axis`, `freq_axis`.

**Acceptance criteria:**
- [ ] Shape da saída: `(n_freqs, n_samples)`.
- [ ] Resolução em frequência configurável via `n_freqs`.
- [ ] Função `hht(signal, fs, ...)` de alto nível que encapsula EMD + HT + espectro e retorna tudo de uma vez.

**Verification:**
- [ ] Plot do espectro para `sinal_1_semruido.csv` mostra concentração de energia em 60 Hz (fundamental).
- [ ] Plot visual aprovado pelo grupo.

**Dependencies:** Task 3

**Files likely touched:**
- `hht/hilbert.py`
- `hht/hht.py`
- `tests/test_hilbert.py`

**Estimated scope:** Medium (3 files)

---

### Checkpoint: Phase 2
- [ ] Todos os testes passam.
- [ ] Espectro de Hilbert plotado e visualmente coerente para ao menos um sinal do projeto.
- [ ] Revisão humana antes de prosseguir.

---

### Phase 3: Detector de Inserção de Cargas Não Lineares

---

#### Task 5: Calcular energia instantânea marginal por IMF

**Description:**  
Implementar `marginal_energy(amplitudes, window_size)` que calcula a energia instantânea suavizada (média móvel do quadrado da amplitude) para cada IMF. Isso fornece um perfil temporal de energia por componente.

**Acceptance criteria:**
- [ ] Retorna array `(n_imfs, n_samples)`.
- [ ] `window_size` configurável (padrão: 1 ciclo da fundamental de 60 Hz → `fs // 60` amostras).
- [ ] Bordas tratadas com modo `'same'`.

**Verification:**
- [ ] Para sinal sem carga não linear, energia é aproximadamente constante.
- [ ] Para sinal com inserção, pico de energia visível no instante de inserção.

**Dependencies:** Task 3

**Files likely touched:**
- `hht/detector.py`
- `tests/test_detector.py`

**Estimated scope:** Small (2 files)

---

#### Task 6: Algoritmo de detecção do instante de inserção

**Description:**  
Implementar `detect_insertion(signal, fs, threshold_factor=3.0, window_size=None)` que:
1. Executa a HHT completa sobre o sinal.
2. Soma a energia marginal das IMFs de alta frequência (IMFs 1 a 3, que capturam harmônicos e transientes).
3. Calcula média `μ` e desvio padrão `σ` da energia no trecho inicial estacionário (primeiros 20% do sinal).
4. Detecta o primeiro instante `t*` onde a energia supera `μ + threshold_factor·σ`.
5. Retorna `t*` em segundos e o índice correspondente.

**Acceptance criteria:**
- [ ] Retorna `(t_seconds, sample_index)` ou `(None, None)` se nenhum evento detectado.
- [ ] `threshold_factor` configurável para ajuste de sensibilidade.
- [ ] Funciona para qualquer comprimento de sinal e taxa de amostragem.

**Verification:**
- [ ] Testar nos 9 pares de sinais (`sinal_N_semruido.csv` vs `sinal_N_ruido.csv`).
- [ ] Comparar instante detectado com inspeção visual do sinal no tempo.
- [ ] Erro de detecção documentado (em ms).

**Dependencies:** Task 4, Task 5

**Files likely touched:**
- `hht/detector.py`
- `tests/test_detector.py`

**Estimated scope:** Medium (3 files)

---

#### Task 7: Script de análise e geração de figuras

**Description:**  
Criar `run_analysis.py` que carrega cada sinal CSV, aplica o detector, gera e salva figuras (sinal no tempo, espectro de Hilbert, energia marginal com linha do instante detectado) em formato vetorial (PDF/SVG) para uso no relatório LaTeX.

**Acceptance criteria:**
- [ ] Figuras salvas em `figures/` com nomes descritivos.
- [ ] Tabela de resultados impressa no terminal (sinal, instante detectado, IMFs usadas).
- [ ] Nenhuma dependência hard-coded de caminho absoluto.

**Verification:**
- [ ] Executar `python run_analysis.py` sem erros.
- [ ] Abrir ao menos 3 figuras e confirmar qualidade visual.

**Dependencies:** Task 6

**Files likely touched:**
- `run_analysis.py`
- `figures/` (diretório de saída)

**Estimated scope:** Medium (2-3 files)

---

### Checkpoint: Phase 3 — Complete
- [ ] Todos os testes passam (`pytest tests/`).
- [ ] Detecção funciona nos 9 sinais do projeto.
- [ ] Figuras geradas e prontas para o relatório.
- [ ] Código da função HHT separado e documentado para inclusão no Apêndice do LaTeX.
- [ ] Revisão final pelo grupo.

---

## Estrutura de Arquivos Proposta

```
Projeto_Final_PDS/
├── hht/
│   ├── __init__.py
│   ├── emd.py          # _sift(), emd()
│   ├── hilbert.py      # hilbert_transform(), hilbert_spectrum()
│   ├── hht.py          # hht()  — função de alto nível
│   └── detector.py     # marginal_energy(), detect_insertion()
├── tests/
│   ├── test_emd.py
│   ├── test_hilbert.py
│   └── test_detector.py
├── sinais/sinais/      # CSVs fornecidos
├── figures/            # saídas vetoriais para o relatório
├── run_analysis.py     # script principal
└── Implemetation_plan_V1.md
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| EMD lenta para sinais longos | Médio | Limitar `max_imfs` e usar interpolação eficiente |
| Efeito de borda na interpolação cúbica | Alto | Adicionar extremos espelhados nas bordas antes do sifting |
| Threshold de detecção sensível ao ruído | Alto | Parâmetro configurável + teste nos sinais com e sem ruído |
| Frequência instantânea instável em amplitudes baixas | Médio | Mascarar frequências onde amplitude < 1% do máximo |

## Premissas Confirmadas

- Frequência fundamental: **60 Hz** (senoidal).
- Todos os sinais têm **1537 amostras** por arquivo CSV (coluna única, sem cabeçalho).

## Open Questions

- Qual a taxa de amostragem (`fs`) dos CSVs? Os arquivos não contêm eixo de tempo. Confirmar com o professor ou inferir pelo enunciado (necessária para normalizar frequência instantânea e calcular `window_size`).
- Existe ground truth do instante de inserção para validação quantitativa?
- Quantas IMFs de alta frequência usar na detecção (1–3 ou mais)? Definir após inspeção visual do espectro de Hilbert.
