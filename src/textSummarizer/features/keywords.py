"""Keyword extraction: TF-IDF term weights and RAKE key phrases."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from textSummarizer.extractive.vectorizer import TfidfVectorizer
from textSummarizer.preprocessing import KEYWORD_STOPWORDS, Document, tokenize

# Phrase delimiters for RAKE. Hyphens and apostrophes stay inside words.
_PHRASE_DELIMITER_RE = re.compile(r"[,.;:!?()\[\]{}\"/|]+|\s-\s")


@dataclass(frozen=True)
class Keyword:
    text: str
    score: float


def extract_tfidf_keywords(
    doc: Document,
    top_k: int = 10,
    stopwords: frozenset[str] = KEYWORD_STOPWORDS,
) -> list[Keyword]:
    """Rank normalized terms by their summed (un-normalized) TF-IDF weight.

    Terms are shown in their most frequent surface form, so the stem "summar"
    is reported as "summarization".
    """
    if len(doc) == 0:
        return []
    tm = TfidfVectorizer(normalize=False).fit_transform(doc.term_lists)
    if tm.matrix.size == 0:
        return []
    totals = tm.matrix.sum(axis=0)

    surfaces: dict[str, Counter] = defaultdict(Counter)
    for sentence in doc.sentences:
        for word, norm, keep in zip(sentence.words, sentence.norms, sentence.is_content, strict=True):
            if keep:
                surfaces[norm][word] += 1

    terms = tm.terms
    ranked = []
    for i in totals.argsort(kind="stable")[::-1]:
        surface = surfaces[terms[i]].most_common(1)[0][0]
        if surface not in stopwords:
            ranked.append((surface, float(totals[i])))
        if len(ranked) == top_k:
            break
    if not ranked:
        return []
    peak = ranked[0][1] or 1.0
    return [Keyword(text=text, score=round(score / peak, 4)) for text, score in ranked]


def extract_rake_keywords(
    doc: Document,
    top_k: int = 10,
    max_words: int = 3,
    stopwords: frozenset[str] = KEYWORD_STOPWORDS,
) -> list[Keyword]:
    """RAKE: Rapid Automatic Keyword Extraction (Rose et al., 2010).

    Candidate phrases are runs of content words between stopwords and
    punctuation. Each word scores ``degree / frequency``, where degree counts
    co-occurrences inside candidate phrases. That favors words that appear in
    longer phrases. A phrase scores the sum of its word scores.
    """
    phrases = [p for sentence in doc.sentences for p in _candidate_phrases(sentence.text, stopwords)]
    phrases = [p for p in phrases if len(p) <= max_words]
    if not phrases:
        return []

    frequency: Counter = Counter()
    degree: Counter = Counter()
    for phrase in phrases:
        for word in phrase:
            frequency[word] += 1
            degree[word] += len(phrase)

    scored: dict[str, float] = {}
    for phrase in phrases:
        key = " ".join(phrase)
        if key not in scored:
            scored[key] = sum(degree[w] / frequency[w] for w in phrase)

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    peak = ranked[0][1] or 1.0
    return [Keyword(text=text, score=round(score / peak, 4)) for text, score in ranked]


def _candidate_phrases(text: str, stopwords: frozenset[str]) -> list[tuple[str, ...]]:
    phrases = []
    for fragment in _PHRASE_DELIMITER_RE.split(text):
        current: list[str] = []
        for word in tokenize(fragment):
            if word in stopwords or word.isdigit() or len(word) < 2:
                if current:
                    phrases.append(tuple(current))
                current = []
            else:
                current.append(word)
        if current:
            phrases.append(tuple(current))
    return phrases
