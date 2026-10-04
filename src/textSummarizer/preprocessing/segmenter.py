"""Sentence segmentation.

Two backends:

* ``"rule"`` (default): regex boundary detection with guards for
  abbreviations, initials, decimals, and lowercase continuations. No external data.
* ``"nltk"``: NLTK's unsupervised Punkt model. Falls back to ``"rule"`` when
  the ``punkt_tab`` data is not installed.
"""

import re
from dataclasses import dataclass

import nltk

from textSummarizer.preprocessing.resources import has_nltk_resource, warn_missing

ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "inc",
        "ltd", "co", "corp", "dept", "univ", "approx", "fig", "gen", "gov",
        "sen", "rep", "lt", "col", "capt", "sgt", "rev", "jan", "feb", "mar",
        "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "e.g",
        "i.e", "a.m", "p.m", "u.s", "u.k", "u.n", "ph.d", "et al", "al",
    }
)  # fmt: skip

# Terminal punctuation, optional closing quotes/brackets, then whitespace.
_BOUNDARY_RE = re.compile(r"([.!?]+)([\"')\]]*)(\s+)")
_PREV_TOKEN_RE = re.compile(r"(\S+)$")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

SEGMENTER_BACKENDS = ("rule", "nltk")


@dataclass
class SentenceSegmenter:
    backend: str = "rule"
    # Each line is one unit (chats, dialogues), so an utterance keeps its speaker.
    split_on_newlines: bool = False

    def __post_init__(self) -> None:
        if self.backend not in SEGMENTER_BACKENDS:
            raise ValueError(f"Unknown segmenter backend {self.backend!r}; use one of {SEGMENTER_BACKENDS}")

    def split(self, text: str) -> list[str]:
        if self.split_on_newlines:
            return [line.strip() for line in text.splitlines() if line.strip()]

        sentences: list[str] = []
        for paragraph in _PARAGRAPH_RE.split(text):
            sentences.extend(self._split_block(paragraph.replace("\n", " ").strip()))
        return [s for s in (s.strip() for s in sentences) if s]

    def _split_block(self, block: str) -> list[str]:
        if not block:
            return []
        if self.backend == "nltk":
            if has_nltk_resource("tokenizers/punkt_tab"):
                return nltk.sent_tokenize(block)
            warn_missing("tokenizers/punkt_tab", "the rule-based segmenter")
        return self._rule_split(block)

    def _rule_split(self, block: str) -> list[str]:
        sentences, start = [], 0
        for match in _BOUNDARY_RE.finditer(block):
            next_char = block[match.end() : match.end() + 1]
            if self._is_boundary(block, match.start(1), match.group(1), next_char):
                sentences.append(block[start : match.end(2)])
                start = match.end()
        sentences.append(block[start:])
        return sentences

    @staticmethod
    def _is_boundary(block: str, punct_pos: int, punct: str, next_char: str) -> bool:
        if next_char.islower():
            return False  # "e.g. this" / "5 p.m. on Monday"
        if punct != ".":
            return True

        prev = _PREV_TOKEN_RE.search(block, max(0, punct_pos - 40), punct_pos)
        if prev is None:
            return True
        token = prev.group(1).lstrip("\"'([").rstrip(".")
        if token.lower() in ABBREVIATIONS:
            return False
        if len(token) == 1 and token.isalpha() and token.isupper():
            return False  # initials: "J. K. Rowling"
        return True
