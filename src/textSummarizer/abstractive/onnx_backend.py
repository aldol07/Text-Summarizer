"""ONNX Runtime implementation of the beam-search backend protocol (no torch needed).

This is what the deployed API uses. It needs only ``numpy`` and
``onnxruntime`` (~15 MB wheel) instead of PyTorch (~700 MB+), which is what
makes the LSTM fit on a 512 MB free-tier instance.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import onnxruntime as ort

from textSummarizer.abstractive.config import LstmConfig
from textSummarizer.abstractive.vocab import Vocabulary

ENCODER_FILE, DECODER_FILE, VOCAB_FILE, CONFIG_FILE = "encoder.onnx", "decoder_step.onnx", "vocab.json", "config.json"


@dataclass
class OnnxDecoderState:
    enc_out: np.ndarray  # (1, L, 2H)
    enc_feat: np.ndarray
    src_ext: np.ndarray  # (L,)
    n_oov: int
    h: np.ndarray  # (N, H)
    c: np.ndarray
    context: np.ndarray  # (N, 2H)
    coverage: np.ndarray  # (N, L)


class OnnxBackend:
    def __init__(self, model_dir: str | Path, use_pointer: bool, threads: int = 1):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        model_dir = Path(model_dir)
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(model_dir / ENCODER_FILE), options, providers=providers)
        self.decoder = ort.InferenceSession(str(model_dir / DECODER_FILE), options, providers=providers)
        self.use_pointer = use_pointer
        self.vocab_size = self.decoder.get_outputs()[0].shape[1]
        self.device = "onnxruntime-cpu"

    def encode(self, src: np.ndarray, src_ext: np.ndarray, n_oov: int) -> OnnxDecoderState:
        enc_out, enc_feat, h0, c0 = self.encoder.run(None, {"src": src.astype(np.int64)})
        return OnnxDecoderState(
            enc_out=enc_out,
            enc_feat=enc_feat,
            src_ext=src_ext[0].astype(np.int64),
            n_oov=n_oov if self.use_pointer else 0,
            h=h0,
            c=c0,
            context=np.zeros((1, enc_out.shape[-1]), dtype=np.float32),
            coverage=np.zeros((1, src.shape[1]), dtype=np.float32),
        )

    def step(self, y_prev: np.ndarray, state: OnnxDecoderState):
        p_vocab, attn, p_gen, h, c, context, coverage = self.decoder.run(
            None,
            {
                "y_prev": y_prev.astype(np.int64),
                "h": state.h,
                "c": state.c,
                "context": state.context,
                "coverage": state.coverage,
                "enc_out": state.enc_out,
                "enc_feat": state.enc_feat,
            },
        )
        if self.use_pointer:
            dist = np.concatenate([p_gen * p_vocab, np.zeros((len(y_prev), state.n_oov), np.float32)], axis=1)
            rows = np.repeat(np.arange(len(y_prev)), attn.shape[1])
            np.add.at(dist, (rows, np.tile(state.src_ext, len(y_prev))), ((1.0 - p_gen) * attn).ravel())
        else:
            dist = p_vocab
        new_state = replace(state, h=h, c=c, context=context, coverage=coverage)
        return dist, p_gen[:, 0], new_state

    def reorder(self, state: OnnxDecoderState, beam_indices: np.ndarray) -> OnnxDecoderState:
        return replace(
            state,
            h=state.h[beam_indices],
            c=state.c[beam_indices],
            context=state.context[beam_indices],
            coverage=state.coverage[beam_indices],
        )


def load_onnx(model_dir: str | Path, threads: int = 1) -> tuple[OnnxBackend, Vocabulary, LstmConfig]:
    model_dir = Path(model_dir)
    config = LstmConfig.from_dict(json.loads((model_dir / CONFIG_FILE).read_text())["config"])
    vocab = Vocabulary.load(model_dir / VOCAB_FILE)
    return OnnxBackend(model_dir, config.model.use_pointer, threads), vocab, config
