"""High-level entry point: one call gives a summary, keywords, stats, and quality metrics.

The CLI and the FastAPI layer both use this, so neither needs to know how the
individual components fit together. Extractive methods are always available.
The abstractive LSTM is optional (pass an ``abstractive`` summarizer, e.g.
``LstmSummarizer.from_onnx(...)``).
"""

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from textSummarizer.evaluation.intrinsic import IntrinsicMetrics, intrinsic_metrics
from textSummarizer.evaluation.rouge import RougeScore, RougeScorer
from textSummarizer.extractive import SUMMARIZERS, ExtractiveSummarizer, SummaryResult, create_summarizer
from textSummarizer.features import (
    Keyword,
    TextStats,
    compression_ratio,
    compute_stats,
    extract_rake_keywords,
    extract_tfidf_keywords,
)
from textSummarizer.preprocessing import Document, Preprocessor, PreprocessorConfig

LSTM_METHOD = "lstm"


@dataclass(frozen=True)
class SelectionOptions:
    num_sentences: int | None = None
    ratio: float | None = None
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    position_weight: float = 0.0
    min_words: int = 0


@dataclass(frozen=True)
class AnalysisResult:
    summary: Any  # SummaryResult (extractive) or AbstractiveResult (lstm)
    kind: str = "extractive"
    keywords: list[Keyword] = field(default_factory=list)
    key_phrases: list[Keyword] = field(default_factory=list)
    stats: TextStats | None = None
    summary_stats: TextStats | None = None
    compression_ratio: float = 0.0
    quality: IntrinsicMetrics | None = None  # reference-free: coverage, redundancy, compression
    rouge: dict[str, RougeScore] | None = None  # only when a reference summary is supplied

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonRow:
    method: str
    kind: str
    summary: str
    summary_words: int
    latency_ms: float
    selected_indices: list[int] = field(default_factory=list)
    rouge: dict[str, RougeScore] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SummarizationService:
    def __init__(self, preprocessor_config: PreprocessorConfig | None = None, abstractive: Any | None = None):
        self.preprocessor = Preprocessor(preprocessor_config)
        self.abstractive = abstractive
        self._summarizers: dict[str, ExtractiveSummarizer] = {}
        self._rouge = RougeScorer()

    @property
    def methods(self) -> list[str]:
        return list(SUMMARIZERS) + ([LSTM_METHOD] if self.abstractive is not None else [])

    def get_summarizer(self, method: str, **params: Any) -> ExtractiveSummarizer:
        if params:  # custom hyperparameters: build a fresh, uncached instance
            return create_summarizer(method, self.preprocessor, **params)
        key = method.lower()
        if key not in self._summarizers:
            self._summarizers[key] = create_summarizer(key, self.preprocessor)
        return self._summarizers[key]

    def analyze(
        self,
        text: str,
        method: str = "textrank",
        options: SelectionOptions | None = None,
        top_keywords: int = 10,
        method_params: dict[str, Any] | None = None,
        reference: str | None = None,
    ) -> AnalysisResult:
        """Summarize ``text``. Pass a human-written ``reference`` summary to also get ROUGE."""
        doc = self.preprocessor.process(text)
        summary, summary_doc, kind = self._run(doc, text, method.lower(), options, method_params)
        stats = compute_stats(doc.text, len(doc))
        summary_stats = compute_stats(summary.summary, len(summary_doc))
        return AnalysisResult(
            summary=summary,
            kind=kind,
            keywords=extract_tfidf_keywords(doc, top_keywords),
            key_phrases=extract_rake_keywords(doc, top_keywords),
            stats=stats,
            summary_stats=summary_stats,
            compression_ratio=compression_ratio(stats.words, summary_stats.words),
            quality=intrinsic_metrics(doc, summary_doc),
            rouge=self._maybe_rouge(reference, summary_doc),
        )

    def compare(
        self,
        text: str,
        methods: list[str] | None = None,
        options: SelectionOptions | None = None,
        reference: str | None = None,
    ) -> list[ComparisonRow]:
        """Run several methods on one document (tokenized once), with latency and optional ROUGE."""
        doc = self.preprocessor.process(text)
        rows = []
        for method in methods or self.methods:
            start = time.perf_counter()
            summary, summary_doc, kind = self._run(doc, text, method.lower(), options, None)
            latency_ms = (time.perf_counter() - start) * 1000
            rows.append(
                ComparisonRow(
                    method=method.lower(),
                    kind=kind,
                    summary=summary.summary,
                    summary_words=summary_doc.word_count,
                    latency_ms=round(latency_ms, 2),
                    selected_indices=getattr(summary, "selected_indices", []),
                    rouge=self._maybe_rouge(reference, summary_doc),
                )
            )
        return rows

    def rouge(self, reference: str, summary_sentences: list[str]) -> dict[str, RougeScore]:
        """ROUGE-1/2/L/Lsum of a summary against a reference (sentences split for Lsum)."""
        reference_lines = "\n".join(self.preprocessor.segmenter.split(self.preprocessor.cleaner.clean(reference)))
        return self._rouge.score(reference_lines, "\n".join(summary_sentences))

    def _maybe_rouge(self, reference: str | None, summary_doc: Document) -> dict[str, RougeScore] | None:
        if not reference or not reference.strip():
            return None
        return self.rouge(reference, [s.text for s in summary_doc.sentences])

    def _run(
        self,
        doc: Document,
        text: str,
        method: str,
        options: SelectionOptions | None,
        method_params: dict[str, Any] | None,
    ) -> tuple[Any, Document, str]:
        """Return (result, summary as a Document, kind)."""
        if method == LSTM_METHOD:
            if self.abstractive is None:
                raise ValueError("The LSTM model is not loaded on this server")
            result = self.abstractive.summarize(text)
            return result, self.preprocessor.process(result.summary), "abstractive"

        summarizer = self.get_summarizer(method, **(method_params or {}))
        result: SummaryResult = summarizer.summarize_document(doc, **asdict(options or SelectionOptions()))
        return result, replace(doc, sentences=[doc.sentences[i] for i in result.selected_indices]), "extractive"
