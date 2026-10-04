"""Command-line interface.

python -m textSummarizer.cli examples/sample_article.txt --method textrank -n 3
python -m textSummarizer.cli examples/sample_article.txt --compare --mmr
cat article.txt | python -m textSummarizer.cli - --json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from textSummarizer.extractive import available_methods
from textSummarizer.preprocessing import PreprocessorConfig
from textSummarizer.service import SelectionOptions, SummarizationService


def build_parser() -> argparse.ArgumentParser:
    methods = [m["name"] for m in available_methods()]
    parser = argparse.ArgumentParser(prog="textSummarizer", description="Classical extractive text summarization")
    parser.add_argument("input", nargs="?", help="Path to a text file, or '-' for stdin")
    parser.add_argument("-m", "--method", default="textrank", choices=methods)

    length = parser.add_mutually_exclusive_group()
    length.add_argument("-n", "--num-sentences", type=int, help="Sentences in the summary (default 3)")
    length.add_argument("-r", "--ratio", type=float, help="Summary size as a fraction of sentences, e.g. 0.3")

    parser.add_argument("--mmr", action="store_true", help="Reduce redundancy with Maximal Marginal Relevance")
    parser.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR relevance/diversity trade-off")
    parser.add_argument("--position-weight", type=float, default=0.0, help="Blend in a lead-position prior (0-1)")
    parser.add_argument("--min-words", type=int, default=0, help="Skip sentences shorter than this")
    parser.add_argument("--normalization", default="stem", choices=["stem", "lemma", "none"])
    parser.add_argument("--segmenter", default="rule", choices=["rule", "nltk"])
    parser.add_argument("--lines", action="store_true", help="Treat each line as a sentence (dialogues, chats)")
    parser.add_argument("--reference", help="File with a human-written summary; prints ROUGE scores")
    parser.add_argument("--compare", action="store_true", help="Run every method side by side")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--list-methods", action="store_true", help="List available methods and exit")
    return parser


def read_input(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_methods:
        for m in available_methods():
            print(f"{m['name']:<10} {m['kind']:<10} {m['description']}")
        return 0
    if not args.input:
        parser.error("input is required (a file path or '-')")

    service = SummarizationService(
        PreprocessorConfig(normalization=args.normalization, segmenter=args.segmenter, split_on_newlines=args.lines)
    )
    options = SelectionOptions(
        num_sentences=args.num_sentences,
        ratio=args.ratio,
        use_mmr=args.mmr,
        mmr_lambda=args.mmr_lambda,
        position_weight=args.position_weight,
        min_words=args.min_words,
    )
    text = read_input(args.input)

    try:
        if args.compare:
            rows = service.compare(text, options=options)
            if args.json:
                print(json.dumps([row.to_dict() for row in rows], indent=2))
            else:
                for row in rows:
                    print(
                        f"\n=== {row.method} (sentences {row.selected_indices}, {row.latency_ms} ms) ===\n{row.summary}"
                    )
            return 0

        start = time.perf_counter()
        reference = read_input(args.reference) if args.reference else None
        analysis = service.analyze(text, method=args.method, options=options, reference=reference)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps({**analysis.to_dict(), "latency_ms": round(elapsed_ms, 2)}, indent=2))
        return 0

    stats = analysis.stats
    print(f"=== Summary ({analysis.summary.method}, {elapsed_ms:.1f} ms) ===")
    print(analysis.summary.summary)
    print("\n=== Keywords ===")
    print(", ".join(k.text for k in analysis.keywords))
    print("\n=== Key phrases ===")
    print(", ".join(k.text for k in analysis.key_phrases))
    print("\n=== Stats ===")
    print(
        f"{stats.words} words, {stats.sentences} sentences, ~{stats.reading_time_seconds}s read, "
        f"Flesch {stats.flesch_reading_ease}, grade {stats.flesch_kincaid_grade}, "
        f"compression {analysis.compression_ratio:.0%}"
    )
    quality = analysis.quality
    print(f"Coverage {quality.coverage:.3f}, redundancy {quality.redundancy:.3f} (reference-free)")
    if analysis.rouge:
        print("\n=== ROUGE vs reference (F1) ===")
        print("  ".join(f"{name} {100 * score.fmeasure:.2f}" for name, score in analysis.rouge.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
