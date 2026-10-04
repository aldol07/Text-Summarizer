"""Locate optional NLTK data packages without forcing a download.

Every NLTK-backed feature has a pure-Python fallback, so missing data only
degrades quality instead of crashing. Run ``python scripts/download_nltk.py``
to install the data into the project-local ``nltk_data/`` directory.
"""

import logging
from functools import cache
from pathlib import Path

import nltk

logger = logging.getLogger(__name__)

PROJECT_NLTK_DIR = Path(__file__).resolve().parents[3] / "nltk_data"


def register_project_nltk_dir() -> None:
    path = str(PROJECT_NLTK_DIR)
    if path not in nltk.data.path:
        nltk.data.path.insert(0, path)


@cache
def has_nltk_resource(resource: str) -> bool:
    """Return True if an NLTK resource such as ``"corpora/wordnet"`` is installed."""
    register_project_nltk_dir()
    for candidate in (resource, f"{resource}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


@cache
def warn_missing(resource: str, fallback: str) -> None:
    """Log a one-time warning when falling back from an NLTK resource."""
    logger.warning(
        "NLTK resource '%s' not found; falling back to %s. Run `python scripts/download_nltk.py` to install it.",
        resource,
        fallback,
    )
