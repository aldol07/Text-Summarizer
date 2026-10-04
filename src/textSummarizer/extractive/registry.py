"""Name -> summarizer lookup used by the CLI, the service layer, and (later) the API."""

from typing import Any

from textSummarizer.extractive.base import ExtractiveSummarizer
from textSummarizer.extractive.baselines import LeadSummarizer, RandomSummarizer
from textSummarizer.extractive.lexrank import LexRankSummarizer
from textSummarizer.extractive.lsa import LsaSummarizer
from textSummarizer.extractive.luhn import LuhnSummarizer
from textSummarizer.extractive.textrank import TextRankSummarizer
from textSummarizer.extractive.tfidf import TfidfCentroidSummarizer
from textSummarizer.preprocessing import Preprocessor

ALGORITHMS: tuple[type[ExtractiveSummarizer], ...] = (
    TextRankSummarizer,
    LexRankSummarizer,
    LsaSummarizer,
    TfidfCentroidSummarizer,
    LuhnSummarizer,
)
BASELINES: tuple[type[ExtractiveSummarizer], ...] = (LeadSummarizer, RandomSummarizer)

SUMMARIZERS: dict[str, type[ExtractiveSummarizer]] = {cls.name: cls for cls in ALGORITHMS + BASELINES}


def available_methods() -> list[dict[str, str]]:
    return [
        {"name": cls.name, "description": cls.description, "kind": "baseline" if cls in BASELINES else "algorithm"}
        for cls in SUMMARIZERS.values()
    ]


def create_summarizer(name: str, preprocessor: Preprocessor | None = None, **params: Any) -> ExtractiveSummarizer:
    try:
        cls = SUMMARIZERS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(SUMMARIZERS)}") from None
    return cls(preprocessor=preprocessor, **params)
