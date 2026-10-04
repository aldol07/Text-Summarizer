"""Display-safe text cleaning.

The cleaned text is what users see in the summary, so this step only removes
noise (URLs, emails, odd whitespace, typographic quotes). It never rewrites
words. Linguistic normalization happens later, at token level.
"""

import re
import unicodedata
from dataclasses import dataclass

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_INLINE_SPACE_RE = re.compile(r"[^\S\n]+")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" *\n *")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([,.;:!?])")

_TYPOGRAPHY = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": " - ",
        "…": "...",
        "\r": "\n",
    }
)


@dataclass
class TextCleaner:
    remove_urls: bool = True
    remove_emails: bool = True

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)
        if self.remove_urls:
            text = _URL_RE.sub(" ", text)
        if self.remove_emails:
            text = _EMAIL_RE.sub(" ", text)

        # Collapse whitespace but keep paragraph breaks; the segmenter uses them.
        text = _INLINE_SPACE_RE.sub(" ", text)
        text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
        text = _MANY_NEWLINES_RE.sub("\n\n", text)
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        return text.strip()
