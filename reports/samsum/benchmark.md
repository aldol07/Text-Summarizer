# Extractive summarization on SAMSum (main)

- **Dataset:** samsum / test
- **Documents:** 819
- **Baseline:** Lead-3
- **Date:** 2026-10-04
- **Machine:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, Python 3.11.9
- **Runtime:** 8 s

ROUGE scores are F1 × 100 (mean over documents, 95% bootstrap CI in brackets for R-1). Δ is the difference from **Lead-3**; `*` marks p < 0.05 (paired bootstrap, one-sided).

| System | R-1 | R-2 | R-L | R-Lsum | Δ R-1 vs baseline | Coverage | Redundancy | Latency ms (mean / p95) | Peak KB |
|---|---|---|---|---|---|---|---|---|---|
| Random | 28.93 [28.0, 29.8] | 7.47 | 22.29 | 27.30 | -2.51 | 0.691 | 0.078 | 0.7 / 1.3 | 47 |
| Lead-3 | 31.44 [30.5, 32.3] | 8.75 | 24.24 | 29.53 | — | 0.689 | 0.085 | 0.6 / 1.2 | 47 |
| Luhn | 30.67 [29.8, 31.6] | 8.48 | 23.77 | 28.74 | -0.77 | 0.714 | 0.131 | 0.7 / 1.4 | 48 |
| TF-IDF centroid | **32.29 [31.4, 33.2]** | 10.35 | 24.70 | 29.90 | +0.85* | 0.773 | 0.098 | 0.7 / 1.4 | 108 |
| LSA | 31.99 [31.1, 32.9] | 9.90 | 24.32 | 29.51 | +0.55 | 0.770 | 0.092 | 0.8 / 1.5 | 128 |
| LexRank | 27.49 [26.6, 28.4] | 6.63 | 21.41 | 25.86 | -3.95 | 0.693 | 0.150 | 1.0 / 1.8 | 131 |
| TextRank | 26.77 [25.9, 27.7] | 6.51 | 21.22 | 25.19 | -4.67 | 0.684 | 0.199 | 0.9 / 1.6 | 62 |
| Oracle | 49.15 [48.4, 49.9] | 19.78 | 38.88 | 45.71 | +17.71* | 0.629 | 0.046 | 1.6 / 3.9 | 86 |

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
