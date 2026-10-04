# Extractive summarization on CNN/DailyMail 3.0.0 (main)

- **Dataset:** cnn_dailymail / test
- **Documents:** 1000 (random sample, seed 42)
- **Baseline:** Lead-3
- **Date:** 2026-10-04
- **Machine:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, Python 3.11.9
- **Runtime:** 67 s

ROUGE scores are F1 × 100 (mean over documents, 95% bootstrap CI in brackets for R-1). Δ is the difference from **Lead-3**; `*` marks p < 0.05 (paired bootstrap, one-sided).

| System | R-1 | R-2 | R-L | R-Lsum | Δ R-1 vs baseline | Coverage | Redundancy | Latency ms (mean / p95) | Peak KB |
|---|---|---|---|---|---|---|---|---|---|
| Random | 29.15 [28.5, 29.8] | 8.51 | 17.93 | 26.17 | -11.43 | 0.469 | 0.043 | 4.7 / 9.3 | 116 |
| Lead-3 | **40.58 [39.9, 41.3]** | 17.70 | 25.14 | 36.64 | — | 0.545 | 0.068 | 4.4 / 9.1 | 116 |
| Luhn | 35.87 [35.3, 36.5] | 14.33 | 23.18 | 32.24 | -4.71 | 0.573 | 0.222 | 4.9 / 10.2 | 130 |
| TF-IDF centroid | 35.47 [34.8, 36.1] | 14.01 | 22.55 | 31.79 | -5.11 | 0.610 | 0.204 | 5.3 / 11.1 | 481 |
| LSA | 33.10 [32.5, 33.7] | 12.51 | 20.94 | 29.44 | -7.49 | 0.598 | 0.177 | 9.9 / 26.5 | 624 |
| LexRank | 35.75 [35.0, 36.4] | 13.46 | 22.40 | 32.10 | -4.83 | 0.587 | 0.155 | 6.3 / 13.4 | 634 |
| TextRank | 36.15 [35.4, 36.8] | 14.43 | 23.18 | 32.48 | -4.43 | 0.608 | 0.201 | 5.8 / 12.5 | 207 |
| Oracle | 57.09 [56.5, 57.7] | 33.19 | 39.62 | 52.71 | +16.51* | 0.505 | 0.062 | 15.8 / 34.7 | 240 |

## Systems

- **Random**: Three random sentences (floor)
- **Lead-3**: First three sentences (standard news baseline)
- **Luhn**: Significant-word clusters (1958)
- **TF-IDF centroid**: Cosine similarity to the document centroid
- **LSA**: SVD topic strength (Steinberger & Ježek)
- **LexRank**: Thresholded cosine graph + PageRank
- **TextRank**: Word-overlap graph + PageRank
- **Oracle**: Greedy ROUGE-maximizing selection (upper bound)

## Notes

- *Oracle* greedily picks the sentences that maximize ROUGE against the reference. It reads the answer, so it is an upper bound for sentence extraction, not a real system.
- *Coverage* (cosine of summary vs document TF-IDF) and *Redundancy* (mean pairwise similarity of summary sentences) are reference-free; lower redundancy is better.
- Latency includes preprocessing and is measured single-threaded on the machine listed above.
