# Extractive summarization on CNN/DailyMail 3.0.0 (ablations)

- **Dataset:** cnn_dailymail / test
- **Documents:** 1000 (random sample, seed 42)
- **Baseline:** Lead-3
- **Date:** 2026-10-04
- **Machine:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, Python 3.11.9
- **Runtime:** 74 s

ROUGE scores are F1 × 100 (mean over documents, 95% bootstrap CI in brackets for R-1). Δ is the difference from **Lead-3**; `*` marks p < 0.05 (paired bootstrap, one-sided).

| System | R-1 | R-2 | R-L | R-Lsum | Δ R-1 vs baseline | Coverage | Redundancy | Latency ms (mean / p95) |
|---|---|---|---|---|---|---|---|---|
| Lead-3 | **40.58 [39.9, 41.3]** | 17.70 | 25.14 | 36.64 | — | 0.545 | 0.068 | 2.7 / 5.2 |
| TextRank | 36.15 [35.4, 36.8] | 14.43 | 23.18 | 32.48 | -4.43 | 0.608 | 0.201 | 3.4 / 7.1 |
| TextRank (cosine) | 36.19 [35.5, 36.9] | 14.43 | 23.37 | 32.55 | -4.39 | 0.597 | 0.226 | 4.1 / 8.2 |
| TextRank + MMR | 36.93 [36.3, 37.6] | 14.80 | 23.28 | 33.23 | -3.65 | 0.628 | 0.134 | 4.4 / 10.1 |
| TextRank + position | 38.12 [37.4, 38.8] | 15.91 | 24.38 | 34.33 | -2.46 | 0.599 | 0.199 | 3.4 / 7.0 |
| TextRank + position + MMR | 39.32 [38.6, 40.0] | 16.71 | 24.87 | 35.49 | -1.27 | 0.620 | 0.125 | 4.4 / 10.1 |
| TextRank (lemma) | 36.03 [35.3, 36.7] | 14.27 | 22.95 | 32.33 | -4.55 | 0.609 | 0.200 | 3.0 / 6.4 |
| TextRank (no normalization) | 36.24 [35.5, 36.9] | 14.50 | 23.32 | 32.59 | -4.34 | 0.592 | 0.189 | 2.7 / 5.7 |
| TextRank (keep stopwords) | 33.82 [33.1, 34.5] | 12.39 | 21.22 | 30.15 | -6.77 | 0.681 | 0.221 | 3.5 / 7.8 |
| LexRank (continuous) | 36.19 [35.5, 36.9] | 14.43 | 23.37 | 32.55 | -4.39 | 0.597 | 0.226 | 3.8 / 8.5 |
| LexRank + position | 37.58 [36.8, 38.2] | 14.97 | 23.79 | 33.87 | -3.01 | 0.585 | 0.169 | 4.0 / 8.2 |
| TF-IDF centroid (cosine) | 35.47 [34.8, 36.1] | 13.83 | 22.94 | 31.90 | -5.12 | 0.587 | 0.233 | 3.1 / 6.4 |
| TF-IDF centroid + position | 37.24 [36.5, 37.9] | 15.39 | 23.63 | 33.46 | -3.34 | 0.602 | 0.200 | 3.1 / 6.3 |
| LSA (TF-IDF, all topics) | 28.66 [28.1, 29.3] | 9.01 | 17.27 | 25.41 | -11.92 | 0.561 | 0.055 | 6.3 / 19.3 |
| TextRank (min 8 words) | 36.12 [35.4, 36.8] | 14.41 | 23.16 | 32.46 | -4.46 | 0.609 | 0.201 | 3.4 / 7.1 |

## Systems

- **TextRank (cosine)**: TF-IDF cosine edges instead of word overlap
- **TextRank + MMR**: Redundancy-aware selection
- **TextRank + position**: 30% lead-position prior blended into scores
- **TextRank (lemma)**: WordNet lemmatization instead of Porter stemming
- **LexRank (continuous)**: Weighted edges instead of a 0.1 threshold
- **TF-IDF centroid (cosine)**: Cosine to centroid instead of MEAD's summed centroid weights
- **LSA (TF-IDF, all topics)**: Sublinear TF-IDF weights, every topic with σ >= 0.5·σmax
- **TextRank (min 8 words)**: Skip very short sentences

## Notes

- *Oracle* greedily picks the sentences that maximize ROUGE against the reference. It reads the answer, so it is an upper bound for sentence extraction, not a real system.
- *Coverage* (cosine of summary vs document TF-IDF) and *Redundancy* (mean pairwise similarity of summary sentences) are reference-free; lower redundancy is better.
- Latency includes preprocessing and is measured single-threaded on the machine listed above.
