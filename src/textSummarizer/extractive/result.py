from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoredSentence:
    index: int
    text: str
    score: float  # relevance in [0, 1] after normalization and position weighting
    selected: bool


@dataclass(frozen=True)
class SummaryResult:
    method: str
    summary: str
    selected_indices: list[int]
    sentences: list[ScoredSentence] = field(default_factory=list)

    @property
    def summary_sentences(self) -> list[str]:
        return [self.sentences[i].text for i in self.selected_indices]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
