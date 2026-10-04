"""Five-stage training pipeline: ingest -> validate -> transform -> train -> evaluate.

Each stage is a plain function so stages can be run, tested, or replaced
independently. ``scripts/train_lstm.py`` chains them.
"""

import json
import logging
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from textSummarizer.abstractive.checkpoint import save_checkpoint
from textSummarizer.abstractive.config import LstmConfig
from textSummarizer.abstractive.data import (
    BucketBatchSampler,
    SummarizationDataset,
    collate,
    encode_example,
)
from textSummarizer.abstractive.model import PointerGenerator
from textSummarizer.abstractive.summarizer import LstmSummarizer
from textSummarizer.abstractive.trainer import Trainer
from textSummarizer.abstractive.vocab import Vocabulary, source_tokens, target_tokens
from textSummarizer.evaluation.datasets import Example, load_dataset
from textSummarizer.evaluation.rouge import ROUGE_TYPES, RougeScorer
from textSummarizer.evaluation.significance import bootstrap_ci
from textSummarizer.preprocessing import SentenceSegmenter

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Stage 1 ---------------------------------------------------------------------
def ingest(config: LstmConfig, limits: dict[str, int | None] | None = None) -> dict[str, list[Example]]:
    limits = limits or {}
    return {split: load_dataset(config.data.dataset, split=split, limit=limits.get(split)) for split in SPLITS}


# Stage 2 ---------------------------------------------------------------------
def validate(splits: dict[str, list[Example]], config: LstmConfig) -> dict[str, dict[str, float]]:
    """Fail fast on empty splits or empty texts and report length statistics."""
    report = {}
    for split, examples in splits.items():
        if not examples:
            raise ValueError(f"split {split!r} is empty")
        empty = [e.id for e in examples if not e.text.strip() or not e.reference.strip()]
        if empty:
            raise ValueError(f"split {split!r} has {len(empty)} empty examples, e.g. {empty[:3]}")
        src = [len(source_tokens(e.text, 10**9)) for e in examples]
        tgt = [len(target_tokens(e.reference, 10**9)) for e in examples]
        report[split] = {
            "examples": len(examples),
            "source_tokens_median": statistics.median(src),
            "target_tokens_median": statistics.median(tgt),
            "source_truncated_pct": round(100 * np.mean([n > config.data.max_source_tokens for n in src]), 2),
            "target_truncated_pct": round(100 * np.mean([n > config.data.max_target_tokens for n in tgt]), 2),
        }
        logger.info("%s: %s", split, report[split])
    return report


# Stage 3 ---------------------------------------------------------------------
def transform(
    splits: dict[str, list[Example]], config: LstmConfig
) -> tuple[Vocabulary, dict[str, SummarizationDataset]]:
    """Build the vocabulary from *training* data only, then encode every split."""
    d = config.data
    train = splits["train"]
    vocab = Vocabulary.build(
        (source_tokens(e.text, d.max_source_tokens) + target_tokens(e.reference, d.max_target_tokens) for e in train),
        max_size=config.model.vocab_size,
        min_freq=d.min_freq,
    )
    logger.info("vocabulary: %d tokens (requested max %d)", len(vocab), config.model.vocab_size)
    datasets = {
        split: SummarizationDataset(
            [encode_example(vocab, e.text, e.reference, d.max_source_tokens, d.max_target_tokens) for e in examples]
        )
        for split, examples in splits.items()
    }
    return vocab, datasets


# Stage 4 ---------------------------------------------------------------------
def train(
    config: LstmConfig,
    vocab: Vocabulary,
    datasets: dict[str, SummarizationDataset],
    model_dir: Path,
    device: torch.device,
) -> list[dict]:
    t = config.training
    set_seed(t.seed)
    model_config = config.model
    if model_config.vocab_size != len(vocab):  # small corpora may yield fewer tokens than requested
        model_config = type(model_config)(**{**asdict(model_config), "vocab_size": len(vocab)})
        config = type(config)(**{**config.__dict__, "model": model_config})
    model = PointerGenerator(model_config)
    logger.info("model: %s parameters, device %s", f"{model.num_parameters():,}", device)

    train_ds, val_ds = datasets["train"], datasets["validation"]
    lengths = [len(e.src) for e in train_ds.examples]
    sampler = BucketBatchSampler(lengths, t.batch_size, max_tokens=t.max_batch_tokens, seed=t.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=t.num_workers)
    val_sampler = BucketBatchSampler([len(e.src) for e in val_ds.examples], t.batch_size, t.max_batch_tokens, seed=0)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate, num_workers=t.num_workers)

    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "training_log.csv"
    log_path.unlink(missing_ok=True)

    def save_best(log) -> None:
        save_checkpoint(model_dir, model, vocab, config, {"best_epoch": log.epoch, "val_loss": log.val_loss})

    history = Trainer(model, t, device, log_path, on_improve=save_best).fit(train_loader, val_loader)
    return [asdict(h) for h in history]


# Stage 5 ---------------------------------------------------------------------
def evaluate(
    model_dir: Path,
    examples: list[Example],
    device: str | None = None,
    out_name: str = "test",
) -> dict:
    """Beam-search decode ``examples`` and score with ROUGE (same scorer as the extractive benchmark)."""
    summarizer = LstmSummarizer.from_checkpoint(model_dir, device)
    scorer = RougeScorer()
    segmenter = SentenceSegmenter()
    scores = {k: [] for k in ROUGE_TYPES}
    latencies, records = [], []
    for example in examples:
        start = time.perf_counter()
        result = summarizer.summarize(example.text)
        latencies.append((time.perf_counter() - start) * 1000)
        prediction = "\n".join(segmenter.split(result.summary))
        for name, score in scorer.score(example.reference, prediction).items():
            scores[name].append(score.fmeasure)
        records.append(
            {
                "id": example.id,
                "prediction": result.summary,
                "reference": example.reference,
                "copy_rate": round(float(np.mean(result.copied)), 4) if result.copied else 0.0,
            }
        )

    metrics = {"examples": len(examples), "device": str(summarizer.backend.device)}
    for name, values in scores.items():
        ci = bootstrap_ci(np.array(values))
        metrics[name] = {"mean": ci.mean, "ci95": [ci.low, ci.high]}
    metrics["latency_ms_mean"] = float(np.mean(latencies))
    metrics["copy_rate_mean"] = float(np.mean([r["copy_rate"] for r in records]))
    metrics["avg_summary_words"] = float(np.mean([len(r["prediction"].split()) for r in records]))

    (model_dir / f"{out_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(model_dir / f"{out_name}_predictions.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics
