"""Typed configuration for the pointer-generator, loaded from ``config/lstm.yaml``."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 15000
    embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.3
    use_pointer: bool = True  # copy mechanism (p_gen); False = plain attention seq2seq
    use_coverage: bool = True  # coverage vector in attention + coverage loss


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "samsum"
    max_source_tokens: int = 400
    max_target_tokens: int = 80
    min_freq: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    max_batch_tokens: int | None = 12000  # cap on batch_size x longest source (GPU memory)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 2.0
    coverage_loss_weight: float = 1.0
    patience: int = 4  # early stopping on validation loss
    seed: int = 42
    num_workers: int = 0


@dataclass(frozen=True)
class DecodingConfig:
    beam_size: int = 4
    min_length: int = 8
    max_length: int = 80
    length_penalty: float = 1.0  # GNMT alpha; 0 = raw log-probability
    block_ngram_repeat: int = 3  # forbid repeating any trigram; 0 = off


@dataclass(frozen=True)
class LstmConfig:
    name: str = "pointer_coverage"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LstmConfig":
        sections = {"model": ModelConfig, "data": DataConfig, "training": TrainingConfig, "decoding": DecodingConfig}
        kwargs: dict[str, Any] = {}
        for key, value in values.items():
            if key in sections:
                kwargs[key] = _build(sections[key], value or {})
            elif key == "name":
                kwargs[key] = value
            else:
                raise ValueError(f"Unknown config section {key!r}")
        return cls(**kwargs)


def load_config(path: str | Path, variant: str | None = None) -> LstmConfig:
    """Load ``defaults`` from the YAML file and deep-merge the chosen ``variants`` entry."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    merged = dict(raw.get("defaults", {}))
    if variant:
        variants = raw.get("variants", {})
        if variant not in variants:
            raise ValueError(f"Unknown variant {variant!r}; available: {sorted(variants)}")
        for section, overrides in (variants[variant] or {}).items():
            merged[section] = {**merged.get(section, {}), **(overrides or {})}
        merged["name"] = variant
    return LstmConfig.from_dict(merged)


def _build(cls, values: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)
