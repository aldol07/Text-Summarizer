"""PyTorch implementation of the beam-search backend protocol."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from textSummarizer.abstractive.model import PointerGenerator, final_distribution


@dataclass
class TorchDecoderState:
    enc_out: torch.Tensor  # (1, L, 2H), shared by all beams
    enc_feat: torch.Tensor  # (1, L, 2H)
    src_mask: torch.Tensor  # (1, L)
    src_ext: torch.Tensor  # (1, L)
    n_oov: int
    h: torch.Tensor  # (n_beams, H)
    c: torch.Tensor
    context: torch.Tensor  # (n_beams, 2H)
    coverage: torch.Tensor  # (n_beams, L)


class TorchBackend:
    def __init__(self, model: PointerGenerator, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocab_size = model.cfg.vocab_size
        self.use_pointer = model.cfg.use_pointer

    @torch.no_grad()
    def encode(self, src: np.ndarray, src_ext: np.ndarray, n_oov: int) -> TorchDecoderState:
        src_t = torch.from_numpy(src).to(self.device)
        enc_out, enc_feat, h, c = self.model.encoder(src_t)
        return TorchDecoderState(
            enc_out=enc_out,
            enc_feat=enc_feat,
            src_mask=torch.ones_like(src_t, dtype=torch.bool),
            src_ext=torch.from_numpy(src_ext).to(self.device),
            n_oov=n_oov if self.use_pointer else 0,
            h=h,
            c=c,
            context=enc_out.new_zeros(1, enc_out.size(-1)),
            coverage=enc_out.new_zeros(1, src.shape[1]),
        )

    @torch.no_grad()
    def step(self, y_prev: np.ndarray, state: TorchDecoderState):
        n = len(y_prev)
        enc_out = state.enc_out.expand(n, -1, -1)
        p_vocab, attn, p_gen, h, c, context, coverage = self.model.decoder(
            torch.from_numpy(y_prev).to(self.device),
            state.h,
            state.c,
            state.context,
            state.coverage,
            enc_out,
            state.enc_feat.expand(n, -1, -1),
            state.src_mask.expand(n, -1),
        )
        dist = final_distribution(p_vocab, attn, p_gen, state.src_ext.expand(n, -1), state.n_oov, self.use_pointer)
        new_state = replace(state, h=h, c=c, context=context, coverage=coverage)
        return dist.cpu().numpy(), p_gen.squeeze(1).cpu().numpy(), new_state

    def reorder(self, state: TorchDecoderState, beam_indices: np.ndarray) -> TorchDecoderState:
        idx = torch.from_numpy(beam_indices).to(self.device)
        return replace(
            state,
            h=state.h.index_select(0, idx),
            c=state.c.index_select(0, idx),
            context=state.context.index_select(0, idx),
            coverage=state.coverage.index_select(0, idx),
        )
