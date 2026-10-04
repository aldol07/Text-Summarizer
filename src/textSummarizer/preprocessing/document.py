"""Data model shared by every summarizer and feature extractor."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sentence:
    """One sentence with aligned token views.

    ``words``, ``norms`` and ``is_content`` have the same length:
    ``words[i]`` is the lowercased surface token, ``norms[i]`` its
    stemmed/lemmatized form, and ``is_content[i]`` is False for stopwords and
    very short tokens.
    """

    index: int
    text: str
    words: tuple[str, ...]
    norms: tuple[str, ...]
    is_content: tuple[bool, ...]

    @property
    def terms(self) -> list[str]:
        """Normalized content terms (the bag of words most algorithms use)."""
        return [n for n, keep in zip(self.norms, self.is_content, strict=True) if keep]

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass(frozen=True)
class Document:
    text: str
    sentences: list[Sentence] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sentences)

    @property
    def term_lists(self) -> list[list[str]]:
        return [s.terms for s in self.sentences]

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.sentences)
