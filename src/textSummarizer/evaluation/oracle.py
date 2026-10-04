"""Greedy extractive oracle: an upper bound for sentence-selection methods.

Following Nallapati et al. (2017), sentences are added one at a time, each
time picking the sentence that most improves ROUGE-1 F + ROUGE-2 F of the
selection against the reference. It stops when no sentence helps or
``max_sentences`` is reached. No extractive method can be expected to beat
this, because the oracle reads the answer.
"""

from collections import Counter

from textSummarizer.evaluation.rouge import RougeScorer, ngrams, rouge_n_from_counts


def greedy_oracle(
    sentences: list[str],
    reference: str,
    max_sentences: int = 3,
    scorer: RougeScorer | None = None,
) -> list[int]:
    scorer = scorer or RougeScorer(("rouge1", "rouge2"))
    ref_tokens = scorer.tokenize(reference)
    ref_uni, ref_bi = ngrams(ref_tokens, 1), ngrams(ref_tokens, 2)

    sent_tokens = [scorer.tokenize(s) for s in sentences]
    sent_uni = [ngrams(t, 1) for t in sent_tokens]
    # Bigrams must not span a sentence boundary; collect them per sentence.
    sent_bi = [ngrams(t, 2) for t in sent_tokens]

    selected: list[int] = []
    sel_uni, sel_bi = Counter(), Counter()
    best_score = 0.0
    while len(selected) < max_sentences:
        best_idx = None
        for i in range(len(sentences)):
            if i in selected or not sent_tokens[i]:
                continue
            score = (
                rouge_n_from_counts(ref_uni, sel_uni + sent_uni[i]).fmeasure
                + rouge_n_from_counts(ref_bi, sel_bi + sent_bi[i]).fmeasure
            )
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx is None:
            break
        selected.append(best_idx)
        sel_uni += sent_uni[best_idx]
        sel_bi += sent_bi[best_idx]
    return sorted(selected)
