# Extractive summarization on SAMSum (abstractive)

- **Dataset:** samsum / test
- **Documents:** 819
- **Baseline:** Lead-3
- **Date:** 2026-10-04
- **Machine:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, Python 3.11.9
- **Runtime:** 614 s

ROUGE scores are F1 × 100 (mean over documents, 95% bootstrap CI in brackets for R-1). Δ is the difference from **Lead-3**; `*` marks p < 0.05 (paired bootstrap, one-sided).

| System | R-1 | R-2 | R-L | R-Lsum | Δ R-1 vs baseline | Coverage | Redundancy | Latency ms (mean / p95) |
|---|---|---|---|---|---|---|---|---|
| Lead-3 | 31.44 [30.5, 32.3] | 8.75 | 24.24 | 29.53 | — | 0.689 | 0.085 | 1.7 / 5.6 |
| TF-IDF centroid | 32.29 [31.4, 33.2] | 10.35 | 24.70 | 29.90 | +0.85* | 0.773 | 0.098 | 1.8 / 5.9 |
| TextRank | 26.77 [25.9, 27.7] | 6.51 | 21.22 | 25.19 | -4.67 | 0.684 | 0.199 | 2.5 / 8.8 |
| LSTM seq2seq + attention | 33.93 [32.8, 35.0] | 12.66 | 28.81 | 30.76 | +2.49* | 0.454 | 0.000 | 200.4 / 649.6 |
| LSTM + pointer | 38.83 [37.6, 40.0] | 16.43 | 33.10 | 34.89 | +7.39* | 0.486 | 0.000 | 179.6 / 557.6 |
| LSTM + pointer + coverage | **39.89 [38.7, 41.0]** | 17.01 | 33.33 | 35.49 | +8.45* | 0.511 | 0.000 | 202.2 / 639.6 |
| LSTM + pointer + coverage (int8) | 39.76 [38.5, 40.9] | 16.78 | 33.14 | 35.34 | +8.32* | 0.511 | 0.000 | 146.8 / 483.5 |
| Oracle | 49.15 [48.4, 49.9] | 19.78 | 38.88 | 45.71 | +17.71* | 0.629 | 0.046 | 3.7 / 13.0 |

## Systems

- **Lead-3**: First three utterances
- **TF-IDF centroid**: Best extractive method on SAMSum
- **LSTM seq2seq + attention**: BiLSTM encoder, Bahdanau attention, no copying
- **LSTM + pointer**: + copy mechanism (p_gen)
- **LSTM + pointer + coverage**: + coverage (full pointer-generator)
- **LSTM + pointer + coverage (int8)**: Deployed model: dynamic int8 quantization
- **Oracle**: Greedy ROUGE-maximizing extraction (upper bound)

## Notes

- *Oracle* greedily picks the sentences that maximize ROUGE against the reference. It reads the answer, so it is an upper bound for sentence extraction, not a real system.
- *Coverage* (cosine of summary vs document TF-IDF) and *Redundancy* (mean pairwise similarity of summary sentences) are reference-free; lower redundancy is better.
- Latency includes preprocessing and is measured single-threaded on the machine listed above.
