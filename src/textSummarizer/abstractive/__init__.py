"""Abstractive summarization: LSTM pointer-generator network (requires PyTorch for training).

Import submodules directly (``textSummarizer.abstractive.model`` etc.) so that
lightweight users, such as ONNX inference, don't pay for importing torch.
"""
