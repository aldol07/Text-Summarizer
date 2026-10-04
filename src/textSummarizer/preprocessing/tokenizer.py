"""Word tokenization and token normalization (stemming / lemmatization)."""

import re

from nltk.stem import PorterStemmer, WordNetLemmatizer

from textSummarizer.preprocessing.resources import has_nltk_resource, warn_missing

# Unicode letters/digits. Apostrophes split tokens ("don't" -> "don", "t"),
# matching how the stopword list is written.
_WORD_RE = re.compile(r"[^\W_]+")

NORMALIZATION_METHODS = ("stem", "lemma", "none")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens."""
    return _WORD_RE.findall(text.lower())


class TokenNormalizer:
    """Map words to a canonical form so "running"/"runs" count as one term.

    ``"lemma"`` uses WordNet (verb then noun pass; noun-first would turn
    "was" into "wa") and falls back to Porter stemming when WordNet data is
    unavailable.
    """

    def __init__(self, method: str = "stem"):
        if method not in NORMALIZATION_METHODS:
            raise ValueError(f"Unknown normalization {method!r}; use one of {NORMALIZATION_METHODS}")
        if method == "lemma" and not has_nltk_resource("corpora/wordnet"):
            warn_missing("corpora/wordnet", "Porter stemming")
            method = "stem"

        self.method = method
        self._cache: dict[str, str] = {}
        self._stemmer = PorterStemmer() if method == "stem" else None
        self._lemmatizer = WordNetLemmatizer() if method == "lemma" else None
        if self._lemmatizer is not None:
            self._lemmatizer.lemmatize("warmup")  # WordNet loads lazily (~seconds); pay it at startup

    def __call__(self, word: str) -> str:
        cached = self._cache.get(word)
        if cached is None:
            cached = self._cache[word] = self._normalize(word)
        return cached

    def _normalize(self, word: str) -> str:
        if self._stemmer is not None:
            return self._stemmer.stem(word)
        if self._lemmatizer is not None:
            return self._lemmatizer.lemmatize(self._lemmatizer.lemmatize(word, "v"), "n")
        return word
