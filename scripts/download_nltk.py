"""Download optional NLTK data into the project-local ``nltk_data/`` directory.

The summarizer works without this data, but it enables:
  * punkt_tab    -> ``--segmenter nltk`` (Punkt sentence tokenizer)
  * wordnet      -> ``--normalization lemma`` (WordNet lemmatizer)
  * omw-1.4      -> multilingual WordNet data the lemmatizer expects

Usage: python scripts/download_nltk.py
"""

from pathlib import Path

import nltk

PACKAGES = ("punkt_tab", "wordnet", "omw-1.4")
TARGET_DIR = Path(__file__).resolve().parents[1] / "nltk_data"


def main() -> None:
    TARGET_DIR.mkdir(exist_ok=True)
    for package in PACKAGES:
        nltk.download(package, download_dir=str(TARGET_DIR), quiet=True, raise_on_error=True)
        print(f"downloaded {package} -> {TARGET_DIR}")


if __name__ == "__main__":
    main()
