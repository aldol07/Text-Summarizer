"""Run summarization experiments over a dataset and aggregate the scores.

An :class:`Experiment` is one fully specified system (method, hyperparameters,
selection options, and preprocessing). Experiments that share a preprocessing
config reuse the same parsed documents, so a suite of 20 systems tokenizes each
article once per config.
"""

import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from textSummarizer.evaluation.datasets import Example
from textSummarizer.evaluation.intrinsic import intrinsic_metrics
from textSummarizer.evaluation.oracle import greedy_oracle
from textSummarizer.evaluation.rouge import ROUGE_TYPES, RougeScorer
from textSummarizer.evaluation.significance import bootstrap_ci, paired_bootstrap_pvalue
from textSummarizer.extractive import create_summarizer
from textSummarizer.preprocessing import Document, Preprocessor, PreprocessorConfig
from textSummarizer.service import SelectionOptions

ORACLE = "oracle"
LSTM = "lstm"  # abstractive; params: model_dir, backend ("onnx" | "torch"), device
METRICS = (*ROUGE_TYPES, "coverage", "redundancy", "compression", "latency_ms")


@dataclass(frozen=True)
class Experiment:
    name: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    selection: SelectionOptions = field(default_factory=SelectionOptions)
    preprocessing: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    description: str = ""


@dataclass(frozen=True)
class AggregateRow:
    experiment: str
    method: str
    description: str
    n: int
    means: dict[str, float]
    ci95: dict[str, tuple[float, float]]
    latency_p95_ms: float
    peak_memory_kb: float | None = None
    delta_vs_baseline: dict[str, float] = field(default_factory=dict)
    p_value_vs_baseline: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    experiments: list[Experiment]
    records: list[dict[str, Any]]
    peak_memory_kb: dict[str, float] = field(default_factory=dict)

    def scores(self, experiment: str, metric: str) -> np.ndarray:
        """Per-document scores in a stable document order (for paired tests)."""
        rows = sorted((r for r in self.records if r["experiment"] == experiment), key=lambda r: r["doc_index"])
        return np.array([r[metric] for r in rows], dtype=np.float64)

    def aggregate(self, baseline: str | None = None, n_resamples: int = 1000) -> list[AggregateRow]:
        rows = []
        for exp in self.experiments:
            means, ci95 = {}, {}
            for metric in METRICS:
                ci = bootstrap_ci(self.scores(exp.name, metric), n_resamples=n_resamples)
                means[metric] = ci.mean
                ci95[metric] = (ci.low, ci.high)

            delta, pvalue = {}, {}
            if baseline and baseline != exp.name and any(e.name == baseline for e in self.experiments):
                for metric in ROUGE_TYPES:
                    system, base = self.scores(exp.name, metric), self.scores(baseline, metric)
                    delta[metric] = float(system.mean() - base.mean())
                    pvalue[metric] = paired_bootstrap_pvalue(system, base, n_resamples=n_resamples)

            latencies = self.scores(exp.name, "latency_ms")
            rows.append(
                AggregateRow(
                    experiment=exp.name,
                    method=exp.method,
                    description=exp.description,
                    n=len(latencies),
                    means=means,
                    ci95=ci95,
                    latency_p95_ms=float(np.percentile(latencies, 95)) if latencies.size else 0.0,
                    peak_memory_kb=self.peak_memory_kb.get(exp.name),
                    delta_vs_baseline=delta,
                    p_value_vs_baseline=pvalue,
                )
            )
        return rows


class BenchmarkRunner:
    def __init__(
        self,
        scorer: RougeScorer | None = None,
        profile_memory_docs: int = 0,
        progress: Callable[[int, int], None] | None = None,
    ):
        self.scorer = scorer or RougeScorer()
        self.profile_memory_docs = profile_memory_docs
        self.progress = progress

    def run(self, examples: Sequence[Example], experiments: Sequence[Experiment]) -> BenchmarkResult:
        names = [e.name for e in experiments]
        if len(set(names)) != len(names):
            raise ValueError("Experiment names must be unique")

        groups: dict[PreprocessorConfig, list[Experiment]] = defaultdict(list)
        for exp in experiments:
            groups[exp.preprocessing].append(exp)

        records: list[dict[str, Any]] = []
        peak_memory: dict[str, float] = {}
        total, done = len(examples) * len(groups), 0
        for config, group in groups.items():
            preprocessor = Preprocessor(config)
            summarizers = {e.name: self._build(e, preprocessor) for e in group if e.method != ORACLE}
            for doc_index, example in enumerate(examples):
                start = time.perf_counter()
                doc = preprocessor.process(example.text)
                prep_ms = (time.perf_counter() - start) * 1000

                for exp in group:
                    start = time.perf_counter()
                    summary_doc, selected = self._summarize(exp, summarizers.get(exp.name), preprocessor, doc, example)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    # Extractive methods need the parsed document; the LSTM reads raw text.
                    latency_ms = elapsed_ms if exp.method == LSTM else prep_ms + elapsed_ms
                    records.append(self._record(exp, doc_index, example, doc, summary_doc, selected, latency_ms))

                done += 1
                if self.progress:
                    self.progress(done, total)

            if self.profile_memory_docs:
                for exp in group:
                    if exp.method != LSTM:  # tracemalloc can't see onnxruntime/torch native allocations
                        summarizer = summarizers.get(exp.name)
                        peak_memory[exp.name] = self._peak_memory_kb(exp, summarizer, preprocessor, examples)

        return BenchmarkResult(experiments=list(experiments), records=records, peak_memory_kb=peak_memory)

    @staticmethod
    def _build(exp: Experiment, preprocessor: Preprocessor):
        if exp.method != LSTM:
            return create_summarizer(exp.method, preprocessor, **exp.params)
        from textSummarizer.abstractive.summarizer import LstmSummarizer

        params = {"backend": "onnx", **exp.params}
        if params["backend"] == "onnx":
            return LstmSummarizer.from_onnx(params["model_dir"])
        return LstmSummarizer.from_checkpoint(params["model_dir"], params.get("device"))

    def _summarize(
        self, exp: Experiment, summarizer, preprocessor: Preprocessor, doc: Document, example: Example
    ) -> tuple[Document, list[int]]:
        """Return the summary as a Document plus selected sentence indices (empty for abstractive)."""
        if exp.method == LSTM:
            return preprocessor.process(summarizer.summarize(example.text).summary), []
        if exp.method == ORACLE:
            k = exp.selection.num_sentences or 3
            selected = greedy_oracle([s.text for s in doc.sentences], example.reference, k, self.scorer)
        else:
            selected = summarizer.summarize_document(doc, **asdict(exp.selection)).selected_indices
        return replace(doc, sentences=[doc.sentences[i] for i in selected]), selected

    def _record(
        self,
        exp: Experiment,
        doc_index: int,
        example: Example,
        doc: Document,
        summary_doc: Document,
        selected: list[int],
        latency_ms: float,
    ) -> dict[str, Any]:
        prediction = "\n".join(s.text for s in summary_doc.sentences)
        rouge = self.scorer.score(example.reference, prediction)
        intrinsic = intrinsic_metrics(doc, summary_doc)
        return {
            "experiment": exp.name,
            "doc_index": doc_index,
            "doc_id": example.id,
            **{name: score.fmeasure for name, score in rouge.items()},
            **intrinsic.to_dict(),
            "latency_ms": latency_ms,
            "doc_sentences": len(doc),
            "summary_words": summary_doc.word_count,
            "selected": selected,
        }

    def _peak_memory_kb(self, exp: Experiment, summarizer, preprocessor: Preprocessor, examples) -> float:
        """Peak Python/numpy allocation while summarizing one document (max over a sample)."""
        peak = 0
        for example in examples[: self.profile_memory_docs]:
            tracemalloc.start()
            try:
                doc = preprocessor.process(example.text)
                self._summarize(exp, summarizer, preprocessor, doc, example)
                peak = max(peak, tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
        return peak / 1024
