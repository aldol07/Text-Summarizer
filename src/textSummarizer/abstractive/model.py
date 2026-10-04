"""Pointer-Generator network with coverage (See, Liu & Manning, 2017).

Encoder:   shared embedding -> single-layer BiLSTM -> encoder states h_i
Attention: e_i = vᵀ tanh(W_h h_i + W_s s_t + w_c c_i + b)          (Bahdanau, + coverage)
Decoder:   x_t = W_x [y_{t-1}; h*_{t-1}] -> LSTMCell -> s_t           (input feeding)
           P_vocab = softmax(V' (V [s_t; h*_t] + b) + b')
Pointer:   p_gen = σ(w_h h*_t + w_s s_t + w_x x_t + b)
           P(w) = p_gen · P_vocab(w) + (1 − p_gen) · Σ_{i: w_i = w} a_i
Coverage:  c_t = Σ_{t' < t} a_{t'};  loss_cov = Σ_i min(a_i, c_i)

The decoder is one explicit step function with plain tensor inputs/outputs,
so training (teacher forcing), beam search, and ONNX export share the code.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from textSummarizer.abstractive.config import ModelConfig
from textSummarizer.abstractive.data import Batch
from textSummarizer.abstractive.vocab import Vocabulary

EPS = 1e-12


class Encoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding.embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.feature_proj = nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=False)  # W_h h_i, precomputed once
        self.reduce_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.reduce_c = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor | None = None):
        embedded = self.dropout(self.embedding(src))
        if src_lens is not None:
            packed = pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h, c) = self.lstm(packed)
            enc_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))
        else:  # single unpadded sequence (inference / ONNX export)
            enc_out, (h, c) = self.lstm(embedded)

        # Final forward + backward states -> initial decoder state.
        h0 = torch.relu(self.reduce_h(torch.cat([h[0], h[1]], dim=-1)))
        c0 = torch.relu(self.reduce_c(torch.cat([c[0], c[1]], dim=-1)))
        return enc_out, self.feature_proj(enc_out), h0, c0


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, use_coverage: bool):
        super().__init__()
        self.use_coverage = use_coverage
        self.state_proj = nn.Linear(2 * hidden_dim, 2 * hidden_dim)  # W_s s_t + b_attn
        self.coverage_proj = nn.Linear(1, 2 * hidden_dim, bias=False) if use_coverage else None
        self.v = nn.Linear(2 * hidden_dim, 1, bias=False)

    def forward(self, state, enc_out, enc_feat, src_mask, coverage):
        energy = enc_feat + self.state_proj(state).unsqueeze(1)
        if self.coverage_proj is not None:
            energy = energy + self.coverage_proj(coverage.unsqueeze(-1))
        scores = self.v(torch.tanh(energy)).squeeze(-1)
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return context, attn


class DecoderStep(nn.Module):
    def __init__(self, embedding: nn.Embedding, cfg: ModelConfig):
        super().__init__()
        e, h = cfg.embedding_dim, cfg.hidden_dim
        self.embedding = embedding
        self.use_pointer = cfg.use_pointer
        self.use_coverage = cfg.use_coverage
        self.dropout = nn.Dropout(cfg.dropout)
        self.input_proj = nn.Linear(2 * h + e, e)
        self.cell = nn.LSTMCell(e, h)
        self.attention = Attention(h, cfg.use_coverage)
        self.out_hidden = nn.Linear(3 * h, h)
        self.out_vocab = nn.Linear(h, cfg.vocab_size)
        self.p_gen_proj = nn.Linear(2 * h + 2 * h + e, 1) if cfg.use_pointer else None

    def forward(self, y_prev, h, c, context_prev, coverage, enc_out, enc_feat, src_mask):
        """One decoding step.

        Returns ``p_vocab (B,V), attn (B,L), p_gen (B,1), h, c, context, next_coverage``.
        ``coverage`` is the sum of *previous* attention (what the coverage loss compares against).
        """
        x = self.input_proj(torch.cat([context_prev, self.dropout(self.embedding(y_prev))], dim=-1))
        h, c = self.cell(x, (h, c))
        state = torch.cat([h, c], dim=-1)
        context, attn = self.attention(state, enc_out, enc_feat, src_mask, coverage)

        logits = self.out_vocab(self.dropout(self.out_hidden(torch.cat([h, context], dim=-1))))
        p_vocab = torch.softmax(logits, dim=-1)
        if self.p_gen_proj is not None:
            p_gen = torch.sigmoid(self.p_gen_proj(torch.cat([context, state, x], dim=-1)))
        else:
            p_gen = torch.ones_like(h[:, :1])
        next_coverage = coverage + attn if self.use_coverage else coverage
        return p_vocab, attn, p_gen, h, c, context, next_coverage


def final_distribution(p_vocab, attn, p_gen, src_ext, n_extra: int, use_pointer: bool = True):
    """Mix generation and copy distributions over the extended vocabulary (B, V + n_extra)."""
    if not use_pointer:  # plain seq2seq: no copy slots exist, src_ext may index past V
        return p_vocab
    dist = p_gen * p_vocab
    if n_extra > 0:
        dist = torch.cat([dist, dist.new_zeros(dist.size(0), n_extra)], dim=-1)
    return dist.scatter_add(1, src_ext, (1.0 - p_gen) * attn)


class PointerGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=Vocabulary.pad_id)
        self.encoder = Encoder(self.embedding, cfg.hidden_dim, cfg.dropout)
        self.decoder = DecoderStep(self.embedding, cfg)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch: Batch, coverage_loss_weight: float = 1.0) -> dict[str, torch.Tensor]:
        """Teacher-forced loss: per-sequence mean NLL (+ coverage loss), averaged over the batch."""
        enc_out, enc_feat, h, c = self.encoder(batch.src, batch.src_lens)
        bsz, src_len = batch.src.shape
        context = enc_out.new_zeros(bsz, enc_out.size(-1))
        coverage = enc_out.new_zeros(bsz, src_len)

        targets = batch.tgt_out
        n_extra = batch.max_oovs if self.cfg.use_pointer else 0
        if not self.cfg.use_pointer:  # without copying, OOV targets can only be <unk>
            targets = targets.masked_fill(targets >= self.cfg.vocab_size, Vocabulary.unk_id)

        step_nll, step_cov = [], []
        for t in range(batch.tgt_in.size(1)):
            p_vocab, attn, p_gen, h, c, context, next_coverage = self.decoder(
                batch.tgt_in[:, t], h, c, context, coverage, enc_out, enc_feat, batch.src_mask
            )
            dist = final_distribution(p_vocab, attn, p_gen, batch.src_ext, n_extra, self.cfg.use_pointer)
            gold = dist.gather(1, targets[:, t : t + 1]).squeeze(1)
            step_nll.append(-torch.log(gold + EPS))
            if self.cfg.use_coverage:
                step_cov.append(torch.minimum(attn, coverage).sum(dim=1))
            coverage = next_coverage

        mask = batch.tgt_mask.float()
        lengths = mask.sum(dim=1)
        nll = ((torch.stack(step_nll, dim=1) * mask).sum(dim=1) / lengths).mean()
        losses = {"nll": nll, "loss": nll}
        if step_cov:
            cov = ((torch.stack(step_cov, dim=1) * mask).sum(dim=1) / lengths).mean()
            losses["coverage"] = cov
            losses["loss"] = nll + coverage_loss_weight * cov
        return losses
