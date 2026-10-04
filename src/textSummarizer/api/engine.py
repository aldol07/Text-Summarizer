"""Request-level orchestration: pick mode/method/defaults, run the service, add warnings."""

import logging
import time
from dataclasses import replace
from typing import Any

from textSummarizer.api.schemas import CompareRequest, LengthOptions, SummarizeRequest
from textSummarizer.api.settings import Settings
from textSummarizer.extractive import available_methods
from textSummarizer.features import looks_like_dialogue
from textSummarizer.preprocessing import PreprocessorConfig
from textSummarizer.service import LSTM_METHOD, SelectionOptions, SummarizationService

logger = logging.getLogger(__name__)

# Defaults chosen from the benchmarks (reports/): on news, a lead-position prior + MMR
# gave +3.2 ROUGE-1 for TextRank. On dialogues neither helps.
MODE_DEFAULTS = {
    "article": {"use_mmr": True, "position_weight": 0.3},
    "dialogue": {"use_mmr": False, "position_weight": 0.0},
}
AUTO_METHOD = {"article": "textrank", "dialogue": LSTM_METHOD}
DIALOGUE_FALLBACK = "tfidf"  # best extractive method on SAMSum
LSTM_DESCRIPTION = "Abstractive BiLSTM pointer-generator with coverage (See et al., 2017), trained on SAMSum chats"


class SummarizerEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lstm = self._load_lstm(settings)
        self.services = {
            "article": SummarizationService(PreprocessorConfig(), self.lstm),
            "dialogue": SummarizationService(PreprocessorConfig(split_on_newlines=True), self.lstm),
        }

    @property
    def abstractive_available(self) -> bool:
        return self.lstm is not None

    def methods(self) -> list[dict[str, str]]:
        methods = [
            {"name": "auto", "kind": "auto", "description": "LSTM for chats, TextRank + position + MMR for articles"}
        ]
        methods += [{**m, "kind": "extractive" if m["kind"] == "algorithm" else m["kind"]} for m in available_methods()]
        if self.abstractive_available:
            methods.append({"name": LSTM_METHOD, "kind": "abstractive", "description": LSTM_DESCRIPTION})
        return methods

    def summarize(self, request: SummarizeRequest) -> dict[str, Any]:
        self._check_length(request.text)
        mode = self._mode(request.mode, request.text)
        method = self._method(request.method, mode)
        options = self._options(request, mode)

        start = time.perf_counter()
        result = self.services[mode].analyze(
            request.text, method=method, options=options, top_keywords=request.top_keywords, reference=request.reference
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            **result.to_dict(),
            "method": method,
            "mode": mode,
            "settings": options.__dict__ if result.kind == "extractive" else {},
            "latency_ms": round(latency_ms, 2),
            "warnings": self._warnings(method, mode, request.text, result),
        }

    def compare(self, request: CompareRequest) -> dict[str, Any]:
        self._check_length(request.text)
        mode = self._mode(request.mode, request.text)
        service = self.services[mode]
        methods = [m.lower() for m in request.methods] if request.methods else service.methods
        unknown = set(methods) - set(service.methods)
        if unknown:
            raise ValueError(f"Unknown or unavailable methods: {sorted(unknown)}")
        options = self._options(request, mode)
        rows = service.compare(request.text, methods=methods, options=options, reference=request.reference)
        return {"mode": mode, "settings": options.__dict__, "rows": [r.to_dict() for r in rows]}

    # -- resolution helpers -------------------------------------------------------------
    def _check_length(self, text: str) -> None:
        if not text.strip():
            raise ValueError("Text is empty")
        if len(text) > self.settings.max_text_chars:
            raise ValueError(f"Text is too long ({len(text):,} chars; limit {self.settings.max_text_chars:,})")

    @staticmethod
    def _mode(requested: str, text: str) -> str:
        if requested != "auto":
            return requested
        return "dialogue" if looks_like_dialogue(text) else "article"

    def _method(self, requested: str, mode: str) -> str:
        method = requested.lower()
        if method == "auto":
            method = AUTO_METHOD[mode]
            if method == LSTM_METHOD and not self.abstractive_available:
                method = DIALOGUE_FALLBACK
        if method not in self.services[mode].methods:
            raise ValueError(f"Unknown or unavailable method {requested!r}; choose from {self.services[mode].methods}")
        return method

    @staticmethod
    def _options(request: LengthOptions, mode: str) -> SelectionOptions:
        defaults = MODE_DEFAULTS[mode]
        options = SelectionOptions(
            num_sentences=request.num_sentences,
            ratio=request.ratio,
            use_mmr=defaults["use_mmr"] if request.use_mmr is None else request.use_mmr,
            mmr_lambda=request.mmr_lambda,
            position_weight=defaults["position_weight"] if request.position_weight is None else request.position_weight,
            min_words=request.min_words,
        )
        if options.num_sentences is None and options.ratio is None and mode == "dialogue":
            options = replace(options, num_sentences=3)
        return options

    def _warnings(self, method: str, mode: str, text: str, result) -> list[str]:
        warnings = []
        if method == LSTM_METHOD:
            # User-facing wording: the UI deliberately doesn't name models.
            if mode == "article":
                warnings.append("The chat summarizer works best on conversations; choose 'Article' for other text.")
            limit = self.lstm.config.data.max_source_tokens
            if len(result.summary.tokens) and _approx_tokens(text) > limit:
                warnings.append("Long chat: only the first part of the conversation was used.")
        elif mode == "article" and result.stats and result.stats.sentences < 3:
            warnings.append("Very short text: add a few more sentences for a meaningful summary.")
        return warnings

    @staticmethod
    def _load_lstm(settings: Settings):
        if not settings.enable_abstractive:
            logger.info("abstractive summarization disabled (ENABLE_ABSTRACTIVE=false)")
            return None
        model_dir = settings.lstm_model_dir
        if not (model_dir / "encoder.onnx").exists():
            logger.warning("LSTM model not found at %s; serving extractive methods only", model_dir)
            return None
        from textSummarizer.abstractive.summarizer import LstmSummarizer

        start = time.perf_counter()
        lstm = LstmSummarizer.from_onnx(model_dir, threads=settings.onnx_threads)
        lstm.summarize("Amanda: Hi Jerry!\nJerry: Hi, how are you?")  # warm up ONNX Runtime
        logger.info("LSTM loaded from %s in %.1fs", model_dir, time.perf_counter() - start)
        return lstm


def _approx_tokens(text: str) -> int:
    from textSummarizer.abstractive.vocab import tokenize

    return len(tokenize(text))
