"""Detect chat/dialogue transcripts ("Name: message" per line)."""

import re

_SPEAKER_LINE_RE = re.compile(r"^\s*[A-Z][\w'.-]*(?: [A-Z][\w'.-]*)?\s*:\s*\S")


def looks_like_dialogue(text: str, min_lines: int = 2, threshold: float = 0.6) -> bool:
    """True when at least ``threshold`` of the non-empty lines start with a speaker label."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < min_lines:
        return False
    speaker_lines = sum(1 for line in lines if _SPEAKER_LINE_RE.match(line))
    return speaker_lines / len(lines) >= threshold
