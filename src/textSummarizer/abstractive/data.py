"""Encode (document, summary) pairs into padded tensor batches."""

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Sampler

from textSummarizer.abstractive.vocab import Vocabulary, source_tokens, target_tokens


@dataclass(frozen=True)
class EncodedExample:
    src: list[int]  # fixed-vocabulary ids (embedding input)
    src_ext: list[int]  # extended ids (copy distribution)
    oovs: list[str]  # source words outside the vocabulary, in extended-id order
    tgt_in: list[int]  # <s> y1 ... yn    (decoder inputs, OOV -> <unk>)
    tgt_out: list[int]  # y1 ... yn </s>  (extended ids: copyable OOVs keep their slot)


def encode_example(vocab: Vocabulary, text: str, summary: str, max_src: int, max_tgt: int) -> EncodedExample:
    src_toks = source_tokens(text, max_src)
    src_ext, oovs = vocab.encode_source(src_toks)
    tgt_toks = target_tokens(summary, max_tgt)
    return EncodedExample(
        src=vocab.encode(src_toks),
        src_ext=src_ext,
        oovs=oovs,
        tgt_in=[vocab.bos_id, *vocab.encode(tgt_toks)],
        tgt_out=[*vocab.encode_target(tgt_toks, oovs), vocab.eos_id],
    )


class SummarizationDataset(Dataset):
    def __init__(self, examples: Sequence[EncodedExample]):
        self.examples = [e for e in examples if e.src]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.examples[idx]


@dataclass
class Batch:
    src: torch.Tensor  # (B, L) long
    src_ext: torch.Tensor  # (B, L) long
    src_lens: torch.Tensor  # (B,) long, on CPU (required by pack_padded_sequence)
    src_mask: torch.Tensor  # (B, L) bool
    tgt_in: torch.Tensor  # (B, T) long
    tgt_out: torch.Tensor  # (B, T) long
    tgt_mask: torch.Tensor  # (B, T) bool
    max_oovs: int
    oovs: list[list[str]]

    def to(self, device: torch.device) -> "Batch":
        for name in ("src", "src_ext", "src_mask", "tgt_in", "tgt_out", "tgt_mask"):
            setattr(self, name, getattr(self, name).to(device, non_blocking=True))
        return self


def collate(examples: Sequence[EncodedExample], pad_id: int = Vocabulary.pad_id) -> Batch:
    def pad(seqs: list[list[int]]) -> torch.Tensor:
        width = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_id] * (width - len(s)) for s in seqs], dtype=torch.long)

    src = pad([e.src for e in examples])
    tgt_in = pad([e.tgt_in for e in examples])
    return Batch(
        src=src,
        src_ext=pad([e.src_ext for e in examples]),
        src_lens=torch.tensor([len(e.src) for e in examples], dtype=torch.long),
        src_mask=src != pad_id,
        tgt_in=tgt_in,
        tgt_out=pad([e.tgt_out for e in examples]),
        tgt_mask=pad([[1] * len(e.tgt_out) for e in examples]).bool(),
        max_oovs=max(len(e.oovs) for e in examples),
        oovs=[e.oovs for e in examples],
    )


class BucketBatchSampler(Sampler[list[int]]):
    """Batches of similar source length (less padding), in shuffled order.

    Indices are shuffled, cut into pools of ``batch_size * pool_factor``, sorted
    by length inside each pool, then grouped into batches, and the batches are
    shuffled. This keeps randomness while cutting wasted LSTM steps on padding.

    A batch closes at ``batch_size`` examples or when ``size * longest source``
    would exceed ``max_tokens``. Attention keeps a (batch, source, 2H) tensor
    per decoder step for backprop, so long dialogues need smaller batches to fit
    in GPU memory. 64 x 400 tokens exhausted an 8 GB card.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        max_tokens: int | None = None,
        pool_factor: int = 50,
        seed: int = 0,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.pool = batch_size * pool_factor
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.pool):
            batches.extend(self._group(sorted(indices[start : start + self.pool], key=self.lengths.__getitem__)))
        rng.shuffle(batches)
        return iter(batches)

    def _group(self, sorted_pool: list[int]) -> list[list[int]]:
        batches, current = [], []
        for idx in sorted_pool:  # ascending length: the newest item is always the longest
            too_many_tokens = self.max_tokens and current and (len(current) + 1) * self.lengths[idx] > self.max_tokens
            if len(current) == self.batch_size or too_many_tokens:
                batches.append(current)
                current = []
            current.append(idx)
        if current:
            batches.append(current)
        return batches

    def __len__(self) -> int:
        """Exact number of batches in the *next* epoch (depends on the shuffle when token-capped)."""
        if not self.max_tokens:
            return (len(self.lengths) + self.batch_size - 1) // self.batch_size
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        return sum(
            len(self._group(sorted(indices[s : s + self.pool], key=self.lengths.__getitem__)))
            for s in range(0, len(indices), self.pool)
        )
