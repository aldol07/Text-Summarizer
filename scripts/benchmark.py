"""Benchmark the summarizers and write reports to ``reports/<dataset>/``.

Examples:
    python scripts/benchmark.py --dataset cnn_dailymail --limit 1000
    python scripts/benchmark.py --dataset cnn_dailymail --suite ablations --limit 500
    python scripts/benchmark.py --dataset samsum
    python scripts/benchmark.py --dataset my_docs.jsonl     # {"id", "text", "summary"} per line
"""

import argparse
import platform
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textSummarizer.evaluation.benchmark import BenchmarkRunner  # noqa: E402
from textSummarizer.evaluation.config import build_experiments, load_benchmark_config  # noqa: E402
from textSummarizer.evaluation.datasets import load_dataset  # noqa: E402
from textSummarizer.evaluation.report import write_reports  # noqa: E402

DATASET_TITLES = {"cnn_dailymail": "CNN/DailyMail 3.0.0", "samsum": "SAMSum"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cnn_dailymail", help="cnn_dailymail | samsum | path/to/file.jsonl")
    parser.add_argument("--split", help="Dataset split (default from config, usually 'test')")
    parser.add_argument("--limit", type=int, help="Number of documents (default from config; 0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--suite", default="main", help="Comma-separated suites from the config, e.g. main,ablations")
    parser.add_argument("--config", default=str(ROOT / "config" / "benchmark.yaml"))
    parser.add_argument("--out", help="Output directory (default reports/<dataset>[_<suite>])")
    parser.add_argument("--profile-memory", type=int, default=20, metavar="N", help="Docs used for peak memory (0=off)")
    parser.add_argument("--resamples", type=int, default=1000, help="Bootstrap resamples for CIs and p-values")
    return parser.parse_args()


def progress_bar(done: int, total: int) -> None:
    width = 30
    filled = int(width * done / total)
    print(f"\r  [{'#' * filled}{'.' * (width - filled)}] {done}/{total}", end="", flush=True)
    if done == total:
        print()


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)
    dataset_key = Path(args.dataset).stem if args.dataset.endswith(".jsonl") else args.dataset
    dataset_cfg = config.get("datasets", {}).get(dataset_key, {})
    split = args.split or dataset_cfg.get("split", "test")
    limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    limit = None if limit == 0 else limit
    suites = [s.strip() for s in args.suite.split(",") if s.strip()]

    print(f"Loading {args.dataset} ({split}, limit={limit or 'all'}) ...")
    examples = load_dataset(args.dataset, split=split, limit=limit, seed=args.seed)
    experiments = build_experiments(config, dataset_key, suites)
    print(f"Running {len(experiments)} systems on {len(examples)} documents")

    start = time.perf_counter()
    runner = BenchmarkRunner(profile_memory_docs=args.profile_memory, progress=progress_bar)
    result = runner.run(examples, experiments)
    elapsed = time.perf_counter() - start

    baseline = config.get("baseline")
    rows = result.aggregate(baseline=baseline, n_resamples=args.resamples)
    suite_label = "+".join(suites)
    meta = {
        "title": f"Extractive summarization on {DATASET_TITLES.get(dataset_key, dataset_key)} ({suite_label})",
        "dataset": f"{args.dataset} / {split}",
        "documents": f"{len(examples)} (random sample, seed {args.seed})" if limit else str(len(examples)),
        "baseline": baseline,
        "date": date.today().isoformat(),
        "machine": f"{platform.processor() or platform.machine()}, Python {platform.python_version()}",
        "runtime": f"{elapsed:.0f} s",
    }
    out_dir = (
        Path(args.out)
        if args.out
        else ROOT / "reports" / (dataset_key if suites == ["main"] else f"{dataset_key}_{suite_label}")
    )
    paths = write_reports(result, rows, out_dir, meta)

    print(f"\nDone in {elapsed:.0f}s. Reports:")
    for name, path in paths.items():
        print(f"  {name:<17} {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print()
    print(paths["markdown"].read_text(encoding="utf-8").split("\n## Systems")[0])


if __name__ == "__main__":
    main()
