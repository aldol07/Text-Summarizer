"""Export a trained pointer-generator to two ONNX graphs for torch-free serving.

* ``encoder.onnx``:      src (1, L) -> enc_out, enc_feat (1, L, 2H), h0, c0 (1, H)
* ``decoder_step.onnx``: one decoding step for N beams over a single document:
      y_prev (N,), h, c (N, H), context (N, 2H), coverage (N, L),
      enc_out, enc_feat (1, L, 2H)
   -> p_vocab (N, V), attn (N, L), p_gen (N, 1), h, c, context, coverage

Mixing in the copy distribution (scatter-add over extended ids) and beam
search run in numpy (see ``onnx_backend.py``), so the serving process needs
only ``onnxruntime`` and ``numpy``.
"""

import shutil
from pathlib import Path

import torch
from torch import nn

from textSummarizer.abstractive.checkpoint import CONFIG_FILE, VOCAB_FILE, load_checkpoint
from textSummarizer.abstractive.model import PointerGenerator

ENCODER_FILE, DECODER_FILE = "encoder.onnx", "decoder_step.onnx"
OPSET = 17


class _EncoderExport(nn.Module):
    def __init__(self, model: PointerGenerator):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, src):
        return self.encoder(src)


class _DecoderStepExport(nn.Module):
    """Wraps the decoder step: expands the single document's encoder states to N beams."""

    def __init__(self, model: PointerGenerator):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, y_prev, h, c, context, coverage, enc_out, enc_feat):
        n = y_prev.shape[0]
        enc_out = enc_out.expand(n, -1, -1)
        enc_feat = enc_feat.expand(n, -1, -1)
        src_mask = torch.ones_like(coverage, dtype=torch.bool)
        return self.decoder(y_prev, h, c, context, coverage, enc_out, enc_feat, src_mask)


def export_onnx(model_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _, config = load_checkpoint(model_dir, "cpu")
    h, n_beams, src_len = config.model.hidden_dim, 2, 7

    src = torch.randint(5, config.model.vocab_size, (1, src_len))
    torch.onnx.export(
        _EncoderExport(model),
        (src,),
        out_dir / ENCODER_FILE,
        input_names=["src"],
        output_names=["enc_out", "enc_feat", "h0", "c0"],
        dynamic_axes={"src": {1: "src_len"}, "enc_out": {1: "src_len"}, "enc_feat": {1: "src_len"}},
        opset_version=OPSET,
        dynamo=False,
    )

    with torch.no_grad():
        enc_out, enc_feat, h0, c0 = model.encoder(src)
    step_inputs = (
        torch.full((n_beams,), 5, dtype=torch.long),
        h0.expand(n_beams, -1).contiguous(),
        c0.expand(n_beams, -1).contiguous(),
        torch.zeros(n_beams, 2 * h),
        torch.zeros(n_beams, src_len),
        enc_out,
        enc_feat,
    )
    beams, length = {0: "beams"}, {0: "beams", 1: "src_len"}
    torch.onnx.export(
        _DecoderStepExport(model),
        step_inputs,
        out_dir / DECODER_FILE,
        input_names=["y_prev", "h", "c", "context", "coverage", "enc_out", "enc_feat"],
        output_names=["p_vocab", "attn", "p_gen", "h_out", "c_out", "context_out", "coverage_out"],
        dynamic_axes={
            "y_prev": beams,
            "h": beams,
            "c": beams,
            "context": beams,
            "coverage": length,
            "enc_out": {1: "src_len"},
            "enc_feat": {1: "src_len"},
            "p_vocab": beams,
            "attn": length,
            "p_gen": beams,
            "h_out": beams,
            "c_out": beams,
            "context_out": beams,
            "coverage_out": length,
        },  # fmt: skip
        opset_version=OPSET,
        dynamo=False,
    )
    for name in (VOCAB_FILE, CONFIG_FILE):
        shutil.copy(model_dir / name, out_dir / name)
    return {"encoder": out_dir / ENCODER_FILE, "decoder": out_dir / DECODER_FILE}


def quantize_onnx(onnx_dir: str | Path, out_dir: str | Path) -> None:
    """Dynamic int8 quantization of weights (activations stay float); ~4x smaller files."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_dir, out_dir = Path(onnx_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (ENCODER_FILE, DECODER_FILE):
        quantize_dynamic(onnx_dir / name, out_dir / name, weight_type=QuantType.QInt8)
    for name in (VOCAB_FILE, CONFIG_FILE):
        shutil.copy(onnx_dir / name, out_dir / name)
