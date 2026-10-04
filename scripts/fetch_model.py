"""Download the trained LSTM (int8 ONNX) at build time, e.g. on Render.

    MODEL_URL=https://github.com/<user>/<repo>/releases/download/v0.1.0/lstm-int8.zip \
    MODEL_SHA256=<sha256> python scripts/fetch_model.py

The zip is extracted to LSTM_MODEL_DIR (default artifacts/onnx/pointer_coverage/int8).
Without MODEL_URL this exits quietly and the API serves extractive methods only.
A wrong checksum fails the build, so a corrupted or tampered model is never deployed.
"""

import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {"encoder.onnx", "decoder_step.onnx", "vocab.json", "config.json"}


def main() -> int:
    url = os.environ.get("MODEL_URL", "").strip()
    if not url:
        print("MODEL_URL not set: skipping model download (extractive methods only)")
        return 0

    target = Path(os.environ.get("LSTM_MODEL_DIR", "artifacts/onnx/pointer_coverage/int8"))
    target = target if target.is_absolute() else ROOT / target
    if REQUIRED_FILES <= {p.name for p in target.glob("*")}:
        print(f"model already present in {target}")
        return 0

    print(f"downloading {url}")
    response = httpx.get(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    data = response.content

    expected = os.environ.get("MODEL_SHA256", "").strip().lower()
    actual = hashlib.sha256(data).hexdigest()
    if expected and actual != expected:
        print(f"checksum mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = {Path(n).name for n in archive.namelist()}
        missing = REQUIRED_FILES - names
        if missing:
            print(f"archive is missing {sorted(missing)}", file=sys.stderr)
            return 1
        target.mkdir(parents=True, exist_ok=True)
        for member in archive.infolist():
            name = Path(member.filename).name  # flatten; never trust paths inside the zip
            if name in REQUIRED_FILES:
                (target / name).write_bytes(archive.read(member))
    print(f"model ({len(data) / 1e6:.1f} MB, sha256 {actual[:12]}...) extracted to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
