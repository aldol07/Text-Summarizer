"""Runtime configuration from environment variables (12-factor style).

| Variable              | Default                                   |
|-----------------------|-------------------------------------------|
| LSTM_MODEL_DIR        | artifacts/onnx/pointer_coverage/int8      |
| ENABLE_ABSTRACTIVE    | true                                      |
| ALLOWED_ORIGINS       | http://localhost:8000,http://127.0.0.1:8000 (comma-separated, "*" allowed) |
| MAX_TEXT_CHARS        | 50000                                     |
| MAX_UPLOAD_MB         | 5                                         |
| SERVE_FRONTEND        | true (serve ./frontend at "/")            |
| ONNX_THREADS          | 1                                         |
"""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    lstm_model_dir: Path = PROJECT_ROOT / "artifacts" / "onnx" / "pointer_coverage" / "int8"
    enable_abstractive: bool = True
    allowed_origins: tuple[str, ...] = ("http://localhost:8000", "http://127.0.0.1:8000")
    max_text_chars: int = 50_000
    max_upload_bytes: int = 5 * 1024 * 1024
    serve_frontend: bool = True
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    onnx_threads: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.environ.get("ALLOWED_ORIGINS")
        return cls(
            lstm_model_dir=_path("LSTM_MODEL_DIR", cls.lstm_model_dir),
            enable_abstractive=_bool("ENABLE_ABSTRACTIVE", True),
            allowed_origins=tuple(o.strip() for o in origins.split(",") if o.strip())
            if origins
            else cls.allowed_origins,
            max_text_chars=int(os.environ.get("MAX_TEXT_CHARS", cls.max_text_chars)),
            max_upload_bytes=int(float(os.environ.get("MAX_UPLOAD_MB", 5)) * 1024 * 1024),
            serve_frontend=_bool("SERVE_FRONTEND", True),
            frontend_dir=_path("FRONTEND_DIR", cls.frontend_dir),
            onnx_threads=int(os.environ.get("ONNX_THREADS", 1)),
        )
