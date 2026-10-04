"""Export trained LSTM variants to ONNX (fp32 + int8), verify parity, measure memory.

    python scripts/export_onnx.py                       # all variants in artifacts/lstm/
    python scripts/export_onnx.py --variant pointer_coverage

For each variant, writes artifacts/onnx/<variant>/{fp32,int8}/ and export_report.json:
* parity: do PyTorch and ONNX produce identical summaries on N test dialogues?
* file sizes
* peak process memory of a *fresh* Python process serving with each backend
  (this is the number that decides whether it fits a 512 MB free instance)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textSummarizer.abstractive.export import export_onnx, quantize_onnx  # noqa: E402
from textSummarizer.abstractive.summarizer import LstmSummarizer  # noqa: E402
from textSummarizer.evaluation.datasets import load_samsum  # noqa: E402

# Runs in a separate interpreter so the measurement includes only what serving needs.
MEMORY_PROBE = r"""
import json, sys, time, psutil
sys.path.insert(0, {src!r})
from textSummarizer.abstractive.summarizer import LstmSummarizer
proc = psutil.Process()
start = proc.memory_info().rss
summarizer = {loader}
loaded = proc.memory_info().rss
texts = json.loads(sys.stdin.read())
t0 = time.perf_counter()
for text in texts:
    summarizer.summarize(text)
latency = (time.perf_counter() - t0) / len(texts) * 1000
info = proc.memory_info()
peak = getattr(info, "peak_wset", None) or getattr(info, "rss")
print(json.dumps({{"baseline_mb": start / 2**20, "after_load_mb": loaded / 2**20,
                  "peak_mb": peak / 2**20, "latency_ms": latency, "torch_imported": "torch" in sys.modules}}))
"""


def measure(loader: str, texts: list[str]) -> dict:
    code = MEMORY_PROBE.format(src=str(ROOT / "src"), loader=loader)
    out = subprocess.run(
        [sys.executable, "-c", code], input=json.dumps(texts), capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.glob("*.onnx")) / 1e6


def export_variant(model_dir: Path, out_root: Path, parity_docs: int) -> dict:
    out = out_root / model_dir.name
    fp32, int8 = out / "fp32", out / "int8"
    t0 = time.perf_counter()
    export_onnx(model_dir, fp32)
    quantize_onnx(fp32, int8)
    print(f"  exported in {time.perf_counter() - t0:.0f}s")

    texts = [e.text for e in load_samsum("test", limit=parity_docs)]
    torch_cpu = LstmSummarizer.from_checkpoint(model_dir, "cpu")
    reference = [torch_cpu.summarize(t).summary for t in texts]
    parity = {}
    for name, path in (("fp32", fp32), ("int8", int8)):
        onnx_model = LstmSummarizer.from_onnx(path)
        same = sum(onnx_model.summarize(t).summary == r for t, r in zip(texts, reference, strict=True))
        parity[name] = f"{same}/{len(texts)} identical to PyTorch"
        print(f"  {name}: {parity[name]}")

    probe_texts = texts[:20]
    memory = {
        "torch_cpu": measure(f"LstmSummarizer.from_checkpoint({str(model_dir)!r}, 'cpu')", probe_texts),
        "onnx_fp32": measure(f"LstmSummarizer.from_onnx({str(fp32)!r})", probe_texts),
        "onnx_int8": measure(f"LstmSummarizer.from_onnx({str(int8)!r})", probe_texts),
    }
    for name, m in memory.items():
        torch_note = "torch imported" if m["torch_imported"] else "no torch"
        print(f"  {name:<10} peak {m['peak_mb']:.0f} MB, {m['latency_ms']:.0f} ms/doc, {torch_note}")

    report = {
        "variant": model_dir.name,
        "parity": parity,
        "size_mb": {
            "pytorch_state_dict": (model_dir / "model.pt").stat().st_size / 1e6,
            "onnx_fp32": dir_size_mb(fp32),
            "onnx_int8": dir_size_mb(int8),
        },
        "memory": memory,
    }
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", help="Only this variant (default: every trained variant)")
    parser.add_argument("--models", default=str(ROOT / "artifacts" / "lstm"))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "onnx"))
    parser.add_argument("--parity-docs", type=int, default=50)
    args = parser.parse_args()

    models = Path(args.models)
    dirs = [models / args.variant] if args.variant else sorted(p.parent for p in models.glob("*/model.pt"))
    for model_dir in dirs:
        print(f"{model_dir.name}:")
        export_variant(model_dir, Path(args.out), args.parity_docs)


if __name__ == "__main__":
    main()
