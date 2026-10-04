"""Descriptive text statistics and readability scores."""

import re
from dataclasses import asdict, dataclass

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_SILENT_SUFFIX_RE = re.compile(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]{1,2}")

WORDS_PER_MINUTE = 238  # average adult silent reading speed for non-fiction


@dataclass(frozen=True)
class TextStats:
    characters: int
    words: int
    sentences: int
    syllables: int
    avg_words_per_sentence: float
    flesch_reading_ease: float  # 0-100, higher = easier
    flesch_kincaid_grade: float  # approximate US school grade
    reading_time_seconds: int

    def to_dict(self) -> dict:
        return asdict(self)


def count_syllables(word: str) -> int:
    """Heuristic English syllable count: vowel groups minus silent endings."""
    word = word.lower().strip("'")
    if len(word) <= 3:
        return 1
    word = _SILENT_SUFFIX_RE.sub("", word)
    word = word.removeprefix("y")
    return max(1, len(_VOWEL_GROUP_RE.findall(word)))


def compute_stats(text: str, sentence_count: int) -> TextStats:
    words = _WORD_RE.findall(text)
    n_words, n_sentences = len(words), max(sentence_count, 0)
    syllables = sum(count_syllables(w) for w in words)

    if n_words and n_sentences:
        wps = n_words / n_sentences
        spw = syllables / n_words
        ease = 206.835 - 1.015 * wps - 84.6 * spw
        grade = 0.39 * wps + 11.8 * spw - 15.59
    else:
        wps = ease = grade = 0.0

    return TextStats(
        characters=len(text),
        words=n_words,
        sentences=n_sentences,
        syllables=syllables,
        avg_words_per_sentence=round(wps, 2),
        flesch_reading_ease=round(ease, 2),
        flesch_kincaid_grade=round(grade, 2),
        reading_time_seconds=round(n_words / WORDS_PER_MINUTE * 60),
    )


def compression_ratio(original_words: int, summary_words: int) -> float:
    """Summary length as a fraction of the original (lower = more compressed)."""
    return round(summary_words / original_words, 4) if original_words else 0.0
