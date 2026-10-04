"""Backend-agnostic beam search over the extended (copy) vocabulary.

A backend implements three calls:

* ``encode(src, src_ext, n_oov) -> state``   for a single document
* ``step(y_prev, state) -> (dist, p_gen, state)``   dist: (n_beams, V + n_oov) probabilities
* ``reorder(state, beam_indices) -> state``   keep and duplicate beams

The PyTorch model and the ONNX Runtime export both implement it, so the
deployed model decodes with exactly the code that was evaluated.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from textSummarizer.abstractive.config import DecodingConfig
from textSummarizer.abstractive.vocab import EOL, Vocabulary


class DecoderBackend(Protocol):
    vocab_size: int

    def encode(self, src: np.ndarray, src_ext: np.ndarray, n_oov: int) -> Any: ...

    def step(self, y_prev: np.ndarray, state: Any) -> tuple[np.ndarray, np.ndarray, Any]: ...

    def reorder(self, state: Any, beam_indices: np.ndarray) -> Any: ...


@dataclass
class Hypothesis:
    tokens: list[int]  # extended ids, starting with <s>
    log_prob: float
    p_gens: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.tokens) - 1

    def score(self, alpha: float) -> float:
        """GNMT length-normalized score: log P / ((5 + |y|) / 6) ** alpha."""
        return self.log_prob / (((5 + max(self.length, 1)) / 6) ** alpha)


def beam_search(
    backend: DecoderBackend,
    vocab: Vocabulary,
    src: list[int],
    src_ext: list[int],
    n_oov: int,
    cfg: DecodingConfig,
) -> Hypothesis:
    vocab_size = backend.vocab_size
    banned = [vocab.pad_id, vocab.unk_id, vocab.bos_id]
    if EOL in vocab:
        banned.append(vocab.stoi[EOL])

    state = backend.encode(np.array([src], dtype=np.int64), np.array([src_ext], dtype=np.int64), n_oov)
    live = [Hypothesis(tokens=[vocab.bos_id], log_prob=0.0)]
    finished: list[Hypothesis] = []

    for step in range(cfg.max_length):
        last = np.array([h.tokens[-1] for h in live], dtype=np.int64)
        last[last >= vocab_size] = vocab.unk_id  # copied OOVs are fed back as <unk>
        dist, p_gen, state = backend.step(last, state)
        log_probs = np.log(np.maximum(dist, 1e-12))

        log_probs[:, banned] = -np.inf
        if step < cfg.min_length:
            log_probs[:, vocab.eos_id] = -np.inf
        if cfg.block_ngram_repeat > 0:
            for row, hyp in enumerate(live):
                blocked = _blocked_tokens(hyp.tokens[1:], cfg.block_ngram_repeat)
                if blocked:
                    log_probs[row, list(blocked)] = -np.inf

        totals = log_probs + np.array([h.log_prob for h in live])[:, None]
        flat = totals.ravel()
        k = min(2 * cfg.beam_size, np.isfinite(flat).sum())
        if k == 0:
            break
        top = np.argpartition(-flat, k - 1)[:k]
        top = top[np.argsort(-flat[top])]

        next_live, keep = [], []
        for flat_idx in top:
            row, token = divmod(int(flat_idx), totals.shape[1])
            parent = live[row]
            hyp = Hypothesis(parent.tokens + [token], float(flat[flat_idx]), parent.p_gens + [float(p_gen[row])])
            if token == vocab.eos_id:
                finished.append(hyp)
            else:
                next_live.append(hyp)
                keep.append(row)
            if len(next_live) == cfg.beam_size:
                break

        if len(finished) >= cfg.beam_size or not next_live:
            break
        live = next_live
        state = backend.reorder(state, np.array(keep, dtype=np.int64))

    candidates = finished or live
    return max(candidates, key=lambda h: h.score(cfg.length_penalty))


def _blocked_tokens(tokens: list[int], n: int) -> set[int]:
    """Tokens that would complete an n-gram already present in ``tokens``."""
    if len(tokens) < n - 1:
        return set()
    prefix = tuple(tokens[len(tokens) - (n - 1) :])
    return {tokens[i + n - 1] for i in range(len(tokens) - n + 1) if tuple(tokens[i : i + n - 1]) == prefix}
