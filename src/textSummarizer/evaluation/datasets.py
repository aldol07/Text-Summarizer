"""Benchmark datasets as lists of :class:`Example`.

* CNN/DailyMail 3.0.0 (news, multi-sentence "highlights" as references)
* SAMSum (chat dialogues with one-to-three-sentence abstractive summaries)
* Any JSONL file with ``{"id", "text", "summary"}`` records (your own docs)

Hugging Face files are fetched with ``huggingface_hub`` and cached in the
standard HF cache. Only the requested split is downloaded. ``pandas`` and
``huggingface_hub`` are evaluation-only dependencies, imported lazily.
"""

import json
import random
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from textSummarizer.preprocessing import SentenceSegmenter

CNN_DM_REPO = "abisee/cnn_dailymail"
CNN_DM_VERSION = "3.0.0"
SAMSUM_REPO = "knkarthick/samsum"
DATASETS = ("cnn_dailymail", "samsum")

# A dateline is a short run of capitalized place-name words ("Rio de Janeiro, Brazil (CNN)").
# Requiring capitals avoids eating a real opening sentence that happens to mention CNN.
# This matches 1066 of the 1067 datelined test articles.
_CNN_DATELINE_RE = re.compile(r"^(?:(?:[A-Z][\w'.&-]*|de|la|el|da|do|du|del|of|al)[ ,]+){0,7}?\(CNN\)\s*(?:--\s*)?")
_DM_BYLINE_RE = re.compile(r"^By \.[^.]{0,100}\. ")
_SAMSUM_ATTACHMENT_RE = re.compile(r"^[^:\n]{1,40}:\s*(?:<file_\w+>\s*)+$", re.MULTILINE)


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    reference: str  # newline-separated sentences (needed for ROUGE-Lsum)
    source: str = ""


def clean_cnn_dailymail(article: str) -> str:
    """Drop the "London (CNN) --" dateline and the rare Daily Mail "By . Name ." byline."""
    return _DM_BYLINE_RE.sub("", _CNN_DATELINE_RE.sub("", article.strip()))


def clean_samsum(dialogue: str) -> str:
    """Remove lines that are only attachments ("Hannah: <file_gif>") and blank lines."""
    dialogue = _SAMSUM_ATTACHMENT_RE.sub("", dialogue.replace("\r\n", "\n"))
    return "\n".join(line for line in dialogue.split("\n") if line.strip())


def split_reference(summary: str) -> str:
    """Put each reference sentence on its own line for ROUGE-Lsum."""
    return "\n".join(SentenceSegmenter().split(summary))


def sample(examples: list[Example], limit: int | None, seed: int = 42) -> list[Example]:
    """Reproducible random subset that keeps the original order."""
    if limit is None or limit >= len(examples):
        return examples
    keep = sorted(random.Random(seed).sample(range(len(examples)), limit))
    return [examples[i] for i in keep]


def load_cnn_dailymail(split: str = "test", limit: int | None = None, seed: int = 42) -> list[Example]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CNN_DM_REPO, f"{CNN_DM_VERSION}/{split}-00000-of-00001.parquet", repo_type="dataset")
    frame = pd.read_parquet(path)
    examples = [
        Example(
            id=row.id, text=clean_cnn_dailymail(row.article), reference=row.highlights.strip(), source="cnn_dailymail"
        )
        for row in frame.itertuples(index=False)
    ]
    return sample(examples, limit, seed)


def load_samsum(split: str = "test", limit: int | None = None, seed: int = 42) -> list[Example]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frame = pd.read_csv(hf_hub_download(SAMSUM_REPO, f"{split}.csv", repo_type="dataset")).dropna()
    examples = [
        Example(
            id=str(row.id),
            text=clean_samsum(row.dialogue),
            reference=split_reference(row.summary),
            source="samsum",
        )
        for row in frame.itertuples(index=False)
    ]
    return sample([e for e in examples if e.text], limit, seed)


def load_jsonl(path: str | Path, limit: int | None = None, seed: int = 42) -> list[Example]:
    """Load your own documents: one ``{"id": ..., "text": ..., "summary": ...}`` object per line."""
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "text" not in record or "summary" not in record:
                raise ValueError(f"{path}:{line_no}: each record needs 'text' and 'summary'")
            examples.append(
                Example(
                    id=str(record.get("id", line_no)),
                    text=record["text"],
                    reference=split_reference(record["summary"]),
                    source=Path(path).stem,
                )
            )
    return sample(examples, limit, seed)


def save_jsonl(examples: Iterable[Example], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            record = asdict(example)
            record["summary"] = record.pop("reference")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_dataset(name: str, split: str = "test", limit: int | None = None, seed: int = 42) -> list[Example]:
    """``name`` is one of :data:`DATASETS` or a path to a ``.jsonl`` file."""
    if name == "cnn_dailymail":
        return load_cnn_dailymail(split, limit, seed)
    if name == "samsum":
        return load_samsum(split, limit, seed)
    if name.endswith(".jsonl"):
        return load_jsonl(name, limit, seed)
    raise ValueError(f"Unknown dataset {name!r}; use one of {DATASETS} or a .jsonl path")
