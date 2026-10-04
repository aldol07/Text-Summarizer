"""Train and evaluate one pointer-generator variant.

    python scripts/train_lstm.py --variant pointer_coverage
    python scripts/train_lstm.py --variant pointer --epochs 2 --limit 500      # quick smoke run
    python scripts/train_lstm.py --variant pointer_coverage --eval-only        # re-score saved model

Outputs go to artifacts/lstm/<variant>/ (git-ignored): model.pt, vocab.json,
config.json, training_log.csv, validation report, test metrics, predictions.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from textSummarizer.abstractive import pipeline  # noqa: E402
from textSummarizer.abstractive.config import load_config  # noqa: E402
from textSummarizer.evaluation.datasets import load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="pointer_coverage")
    parser.add_argument("--config", default=str(ROOT / "config" / "lstm.yaml"))
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--limit", type=int, help="Use only N examples per split (smoke tests)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-root", default=str(ROOT / "artifacts" / "lstm"))
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate the saved model")
    return parser.parse_args()


def stage(name: str):
    logging.info("=" * 20 + f" stage: {name} " + "=" * 20)
    return time.perf_counter()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    config = load_config(args.config, args.variant)
    if args.epochs:
        config = replace(config, training=replace(config.training, epochs=args.epochs))
    model_dir = Path(args.out_root) / args.variant
    limits = dict.fromkeys(pipeline.SPLITS, args.limit)
    device = torch.device(args.device)

    if not args.eval_only:
        stage("1/5 data ingestion")
        splits = pipeline.ingest(config, limits)

        stage("2/5 data validation")
        report = pipeline.validate(splits, config)

        stage("3/5 data transformation")
        vocab, datasets = pipeline.transform(splits, config)

        stage("4/5 model training")
        start = time.perf_counter()
        history = pipeline.train(config, vocab, datasets, model_dir, device)
        (model_dir / "data_report.json").write_text(json.dumps(report, indent=2))
        logging.info("training finished in %.1f min, %d epochs", (time.perf_counter() - start) / 60, len(history))
        test_examples = splits["test"]
    else:
        test_examples = load_dataset(config.data.dataset, split="test", limit=args.limit)

    stage("5/5 model evaluation")
    metrics = pipeline.evaluate(model_dir, test_examples, str(device))
    logging.info(
        "test ROUGE-1 %.2f  ROUGE-2 %.2f  ROUGE-L %.2f  ROUGE-Lsum %.2f  (%d docs, %.0f ms/doc)",
        *(100 * metrics[k]["mean"] for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")),
        metrics["examples"],
        metrics["latency_ms_mean"],
    )
    logging.info("artifacts in %s", model_dir)


if __name__ == "__main__":
    main()
