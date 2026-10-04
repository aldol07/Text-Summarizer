"""Request models. Validation errors become 422 responses automatically."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Mode = Literal["auto", "article", "dialogue"]


class LengthOptions(BaseModel):
    num_sentences: int | None = Field(None, ge=1, le=50, description="Sentences to extract")
    ratio: float | None = Field(None, gt=0, le=1, description="Or: fraction of sentences to keep")
    use_mmr: bool | None = Field(None, description="Redundancy control; default depends on the mode")
    mmr_lambda: float = Field(0.7, ge=0, le=1)
    position_weight: float | None = Field(None, ge=0, le=1, description="Lead-position prior; default by mode")
    min_words: int = Field(0, ge=0, le=100)

    @model_validator(mode="after")
    def _one_length(self):
        if self.num_sentences is not None and self.ratio is not None:
            raise ValueError("Pass either num_sentences or ratio, not both")
        return self


class SummarizeRequest(LengthOptions):
    text: str = Field(min_length=1)
    method: str = Field("auto", description="auto | textrank | lexrank | lsa | tfidf | luhn | lead | random | lstm")
    mode: Mode = "auto"
    reference: str | None = Field(None, max_length=20_000, description="Human summary; enables ROUGE")
    top_keywords: int = Field(10, ge=0, le=50)


class CompareRequest(LengthOptions):
    text: str = Field(min_length=1)
    mode: Mode = "auto"
    methods: list[str] | None = Field(None, max_length=10)
    reference: str | None = Field(None, max_length=20_000)


class UrlRequest(BaseModel):
    url: str = Field(min_length=4, max_length=2048)
