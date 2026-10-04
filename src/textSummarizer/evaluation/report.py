"""Write benchmark results as Markdown, CSV, JSON, and a PNG chart."""

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from textSummarizer.evaluation.benchmark import AggregateRow, BenchmarkResult

logger = logging.getLogger(__name__)

ROUGE_COLUMNS = (("rouge1", "R-1"), ("rouge2", "R-2"), ("rougeL", "R-L"), ("rougeLsum", "R-Lsum"))


def write_reports(
    result: BenchmarkResult,
    rows: list[AggregateRow],
    out_dir: str | Path,
    meta: dict[str, Any],
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / "benchmark.md",
        "summary_csv": out / "summary.csv",
        "summary_json": out / "summary.json",
        "per_document_csv": out / "per_document.csv",
    }
    paths["markdown"].write_text(render_markdown(rows, meta), encoding="utf-8")
    _write_summary_csv(rows, paths["summary_csv"])
    paths["summary_json"].write_text(json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=2))
    _write_per_document_csv(result.records, paths["per_document_csv"])
    chart = plot_rouge_chart(rows, out / "rouge_chart.png", title=meta.get("title", "ROUGE F1"))
    if chart:
        paths["chart"] = chart
    return paths


def render_markdown(rows: list[AggregateRow], meta: dict[str, Any]) -> str:
    baseline = meta.get("baseline")
    lines = [f"# {meta.get('title', 'Benchmark')}", ""]
    lines += [f"- **{k.replace('_', ' ').capitalize()}:** {v}" for k, v in meta.items() if k != "title"]
    lines += [
        "",
        "ROUGE scores are F1 × 100 (mean over documents, 95% bootstrap CI in brackets for R-1). "
        f"Δ is the difference from **{baseline}**; `*` marks p < 0.05 (paired bootstrap, one-sided).",
        "",
        "| System | R-1 | R-2 | R-L | R-Lsum | Δ R-1 vs baseline | Coverage | Redundancy | Latency ms (mean / p95) |"
        + (" Peak KB |" if any(r.peak_memory_kb is not None for r in rows) else ""),
        "|---|---|---|---|---|---|---|---|---|" + ("---|" if any(r.peak_memory_kb is not None for r in rows) else ""),
    ]
    best = max((r.means["rouge1"] for r in rows if r.method != "oracle"), default=None)
    for r in rows:
        low, high = r.ci95["rouge1"]
        r1 = f"{100 * r.means['rouge1']:.2f} [{100 * low:.1f}, {100 * high:.1f}]"
        if r.means["rouge1"] == best:
            r1 = f"**{r1}**"
        cells = [r.experiment, r1] + [f"{100 * r.means[m]:.2f}" for m, _ in ROUGE_COLUMNS[1:]]
        if "rouge1" in r.delta_vs_baseline:
            star = "*" if r.p_value_vs_baseline["rouge1"] < 0.05 else ""
            cells.append(f"{100 * r.delta_vs_baseline['rouge1']:+.2f}{star}")
        else:
            cells.append("—")
        cells += [
            f"{r.means['coverage']:.3f}",
            f"{r.means['redundancy']:.3f}",
            f"{r.means['latency_ms']:.1f} / {r.latency_p95_ms:.1f}",
        ]
        if r.peak_memory_kb is not None:
            cells.append(f"{r.peak_memory_kb:.0f}")
        lines.append("| " + " | ".join(cells) + " |")

    descriptions = [(r.experiment, r.description) for r in rows if r.description]
    if descriptions:
        lines += ["", "## Systems", ""] + [f"- **{name}**: {desc}" for name, desc in descriptions]
    lines += [
        "",
        "## Notes",
        "",
        "- *Oracle* greedily picks the sentences that maximize ROUGE against the reference. It reads the "
        "answer, so it is an upper bound for sentence extraction, not a real system.",
        "- *Coverage* (cosine of summary vs document TF-IDF) and *Redundancy* (mean pairwise similarity "
        "of summary sentences) are reference-free; lower redundancy is better.",
        "- Latency includes preprocessing and is measured single-threaded on the machine listed above.",
        "",
    ]
    return "\n".join(lines)


def plot_rouge_chart(rows: list[AggregateRow], path: Path, title: str = "ROUGE F1") -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart")
        return None

    names = [r.experiment for r in rows]
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(names)), 4.5))
    for offset, (metric, label) in zip((-width, 0, width), ROUGE_COLUMNS[:3], strict=True):
        means = np.array([100 * r.means[metric] for r in rows])
        lows = np.array([100 * r.ci95[metric][0] for r in rows])
        highs = np.array([100 * r.ci95[metric][1] for r in rows])
        ax.bar(x + offset, means, width, label=label, yerr=[means - lows, highs - means], capsize=2)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("F1 × 100")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_summary_csv(rows: list[AggregateRow], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        metrics = list(rows[0].means) if rows else []
        writer.writerow(
            [
                "experiment",
                "method",
                "n",
                *metrics,
                "rouge1_ci_low",
                "rouge1_ci_high",
                "latency_p95_ms",
                "peak_memory_kb",
            ]
            + ["delta_rouge1", "p_value_rouge1"]
        )
        for r in rows:
            writer.writerow(
                [r.experiment, r.method, r.n, *(round(r.means[m], 6) for m in metrics)]
                + [round(r.ci95["rouge1"][0], 6), round(r.ci95["rouge1"][1], 6), round(r.latency_p95_ms, 3)]
                + [r.peak_memory_kb, r.delta_vs_baseline.get("rouge1"), r.p_value_vs_baseline.get("rouge1")]
            )


def _write_per_document_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "selected": " ".join(map(str, record["selected"]))})
