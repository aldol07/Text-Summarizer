"""User-facing abstractive summarizer: text in, generated summary out."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from textSummarizer.abstractive.beam_search import DecoderBackend, beam_search
from textSummarizer.abstractive.config import DecodingConfig, LstmConfig
from textSummarizer.abstractive.vocab import Vocabulary, casing_map, detokenize, source_tokens

COPY_THRESHOLD = 0.5  # p_gen below this means the token was mostly copied from the source


@dataclass(frozen=True)
class AbstractiveResult:
    method: str
    summary: str
    tokens: list[str]
    p_gen: list[float]  # per output token: 1 = generated from vocabulary, 0 = copied from source
    copied: list[bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LstmSummarizer:
    name = "lstm"
    description = "BiLSTM pointer-generator with coverage (See et al., 2017), trained on SAMSum"

    def __init__(self, backend: DecoderBackend, vocab: Vocabulary, config: LstmConfig):
        self.backend = backend
        self.vocab = vocab
        self.config = config

    @classmethod
    def from_checkpoint(cls, model_dir: str | Path, device: str | None = None) -> "LstmSummarizer":
        import torch

        from textSummarizer.abstractive.checkpoint import load_checkpoint
        from textSummarizer.abstractive.torch_backend import TorchBackend

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, vocab, config = load_checkpoint(model_dir, device)
        return cls(TorchBackend(model, device), vocab, config)

    @classmethod
    def from_onnx(cls, model_dir: str | Path, threads: int = 1) -> "LstmSummarizer":
        """Load an exported model with ONNX Runtime (no PyTorch import)."""
        from textSummarizer.abstractive.onnx_backend import load_onnx

        backend, vocab, config = load_onnx(model_dir, threads)
        return cls(backend, vocab, config)

    def summarize(self, text: str, **decoding: Any) -> AbstractiveResult:
        """Generate a summary. Keyword arguments override :class:`DecodingConfig` fields."""
        cfg: DecodingConfig = replace(self.config.decoding, **decoding) if decoding else self.config.decoding
        tokens = source_tokens(text, self.config.data.max_source_tokens)
        if not tokens:
            return AbstractiveResult(self.name, "", [], [], [])

        src = self.vocab.encode(tokens)
        src_ext, oovs = self.vocab.encode_source(tokens)
        best = beam_search(self.backend, self.vocab, src, src_ext, len(oovs), cfg)

        ids = best.tokens[1:]
        p_gens = best.p_gens
        if ids and ids[-1] == self.vocab.eos_id:
            ids, p_gens = ids[:-1], p_gens[:-1]
        out_tokens = self.vocab.decode(ids, oovs)
        copied = [i >= len(self.vocab) or p < COPY_THRESHOLD for i, p in zip(ids, p_gens, strict=True)]
        return AbstractiveResult(
            method=self.name,
            summary=detokenize(out_tokens, casing_map(text)),
            tokens=out_tokens,
            p_gen=[round(p, 4) for p in p_gens],
            copied=copied,
        )
