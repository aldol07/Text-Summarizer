"""Raw text -> :class:`Document` (clean, segment, tokenize, normalize, filter)."""

from dataclasses import dataclass, field

from textSummarizer.preprocessing.cleaner import TextCleaner
from textSummarizer.preprocessing.document import Document, Sentence
from textSummarizer.preprocessing.segmenter import SentenceSegmenter
from textSummarizer.preprocessing.stopwords import ENGLISH_STOPWORDS
from textSummarizer.preprocessing.tokenizer import TokenNormalizer, tokenize


@dataclass(frozen=True)
class PreprocessorConfig:
    normalization: str = "stem"  # "stem" | "lemma" | "none"
    segmenter: str = "rule"  # "rule" | "nltk"
    split_on_newlines: bool = False
    remove_stopwords: bool = True
    min_token_length: int = 2
    remove_urls: bool = True
    remove_emails: bool = True
    extra_stopwords: frozenset[str] = field(default_factory=frozenset)


class Preprocessor:
    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()
        self.cleaner = TextCleaner(
            remove_urls=self.config.remove_urls,
            remove_emails=self.config.remove_emails,
        )
        self.segmenter = SentenceSegmenter(
            backend=self.config.segmenter,
            split_on_newlines=self.config.split_on_newlines,
        )
        self.normalizer = TokenNormalizer(self.config.normalization)
        self.stopwords = (
            ENGLISH_STOPWORDS | {w.lower() for w in self.config.extra_stopwords}
            if self.config.remove_stopwords
            else frozenset()
        )

    def process(self, text: str) -> Document:
        cleaned = self.cleaner.clean(text or "")
        sentences = []
        for sentence_text in self.segmenter.split(cleaned):
            words = tokenize(sentence_text)
            if not words:
                continue  # punctuation-only fragments
            sentences.append(
                Sentence(
                    index=len(sentences),
                    text=sentence_text,
                    words=tuple(words),
                    norms=tuple(self.normalizer(w) for w in words),
                    is_content=tuple(self._is_content(w) for w in words),
                )
            )
        return Document(text=cleaned, sentences=sentences)

    def _is_content(self, word: str) -> bool:
        return len(word) >= self.config.min_token_length and word not in self.stopwords
