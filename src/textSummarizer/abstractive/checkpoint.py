"""Save/load a trained model directory: ``model.pt``, ``vocab.json``, ``config.json``."""

import json
from pathlib import Path
from typing import Any

import torch

from textSummarizer.abstractive.config import LstmConfig
from textSummarizer.abstractive.model import PointerGenerator
from textSummarizer.abstractive.vocab import Vocabulary

MODEL_FILE, VOCAB_FILE, CONFIG_FILE = "model.pt", "vocab.json", "config.json"


def save_checkpoint(
    model_dir: str | Path,
    model: PointerGenerator,
    vocab: Vocabulary,
    config: LstmConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / MODEL_FILE)
    vocab.save(out / VOCAB_FILE)
    (out / CONFIG_FILE).write_text(json.dumps({"config": config.to_dict(), **(extra or {})}, indent=2))


def load_checkpoint(model_dir: str | Path, device: str | torch.device = "cpu"):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / CONFIG_FILE).read_text())
    config = LstmConfig.from_dict(meta["config"])
    vocab = Vocabulary.load(model_dir / VOCAB_FILE)
    model = PointerGenerator(config.model)
    model.load_state_dict(torch.load(model_dir / MODEL_FILE, map_location=device, weights_only=True))
    return model.to(device).eval(), vocab, config
