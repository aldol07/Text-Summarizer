"""Seq2seq tokenization, vocabulary, and pointer-generator OOV handling.

Unlike the extractive pipeline (bag of stemmed content words), the decoder
must *generate* readable text, so tokens keep punctuation and contractions:
"I'll bring it!" -> ["i'll", "bring", "it", "!"]. Line breaks (dialogue turns)
become an ``<eol>`` token so the encoder sees speaker boundaries.

Extended vocabulary (See et al., 2017): source words missing from the fixed
vocabulary get temporary ids ``len(vocab) + k`` per example. That lets the
copy mechanism emit them, e.g. rare names.
"""

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from textSummarizer.preprocessing import TextCleaner

PAD, UNK, BOS, EOS, EOL = "<pad>", "<unk>", "<s>", "</s>", "<eol>"
SPECIAL_TOKENS = (PAD, UNK, BOS, EOS, EOL)

_TOKEN_RE = re.compile(r"<eol>|[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
_SPACE_BEFORE_RE = re.compile(r" ([.,!?;:%)\]}])")
_SPACE_AFTER_RE = re.compile(r"([(\[{$]) ")
_SENTENCE_START_RE = re.compile(r"(^|[.!?] )([a-z])")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("\r", "").replace("\n", " <eol> "))


_cleaner = TextCleaner()


def source_tokens(text: str, max_tokens: int) -> list[str]:
    """Cleaning + tokenization shared by training and inference (must stay identical)."""
    return tokenize(_cleaner.clean(text))[:max_tokens]


def target_tokens(summary: str, max_tokens: int) -> list[str]:
    # References are newline-separated sentences (for ROUGE-Lsum); the decoder writes running text.
    return tokenize(summary.replace("\n", " "))[:max_tokens]


def casing_map(text: str) -> dict[str, str]:
    """Lowercase token -> capitalized form, for words that are evidently *names*.

    The model writes lowercase, and this restores names. A word counts as a name if it
    is a speaker label ("Amanda: ...") or is capitalized mid-sentence and never
    appears in lowercase. Sentence-initial words ("And so ...") and SHOUTING /
    acronyms ("MACHINE", "OK") are ignored, so they don't leak into the summary.
    """
    names = {part.lower(): part for label in _SPEAKER_RE.findall(text) for part in label.split()}
    forms: dict[str, Counter] = {}
    lowercase_seen: set[str] = set()
    for match in _WORD_RE.finditer(text):
        word, lower = match.group(), match.group().lower()
        if word == lower:
            lowercase_seen.add(lower)
        elif not (word.isupper() and len(word) > 1) and not _is_sentence_start(text, match.start()):
            forms.setdefault(lower, Counter())[word] += 1
    found = {lower: c.most_common(1)[0][0] for lower, c in forms.items() if lower not in lowercase_seen}
    return {**found, **names}


_SPEAKER_RE = re.compile(r"^[ \t]*([A-Z][\w'.-]*(?: [A-Z][\w'.-]*)?)[ \t]*:", re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-z][\w']*")


def _is_sentence_start(text: str, pos: int) -> bool:
    prefix = text[:pos].rstrip(" \t\"'(")
    return not prefix or prefix[-1] in ".!?:\n"


def detokenize(tokens: Sequence[str], casing: dict[str, str] | None = None) -> str:
    """Join tokens into readable text: fix punctuation spacing, restore casing."""
    casing = casing or {}
    words = []
    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue
        if token == "i" or token.startswith("i'"):
            token = "I" + token[1:]
        words.append(casing.get(token, token))
    text = " ".join(words)
    text = _SPACE_BEFORE_RE.sub(r"\1", text)
    text = _SPACE_AFTER_RE.sub(r"\1", text)
    text = re.sub(r" (n't|'s|'re|'ll|'ve|'d|'m)\b", r"\1", text)
    text = re.sub(r"(\d)([:.,]) (\d)", r"\1\2\3", text)  # "7 : 45" -> "7:45", "3 . 5" -> "3.5"
    return _SENTENCE_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), text)


class Vocabulary:
    def __init__(self, tokens: Iterable[str]):
        self.itos: list[str] = list(SPECIAL_TOKENS) + [t for t in tokens if t not in SPECIAL_TOKENS]
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    pad_id, unk_id, bos_id, eos_id = 0, 1, 2, 3

    @classmethod
    def build(cls, texts: Iterable[Sequence[str]], max_size: int = 15000, min_freq: int = 3) -> "Vocabulary":
        counts = Counter(t for tokens in texts for t in tokens)
        ranked = [t for t, c in counts.most_common() if c >= min_freq and t not in SPECIAL_TOKENS]
        return cls(ranked[: max_size - len(SPECIAL_TOKENS)])

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def encode(self, tokens: Sequence[str]) -> list[int]:
        """Fixed-vocabulary ids (OOV -> <unk>); used as embedding inputs."""
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def encode_source(self, tokens: Sequence[str]) -> tuple[list[int], list[str]]:
        """Extended ids for the copy distribution, plus this example's OOV list."""
        ids, oovs = [], []
        for token in tokens:
            idx = self.stoi.get(token)
            if idx is None:
                if token not in oovs:
                    oovs.append(token)
                idx = len(self) + oovs.index(token)
            ids.append(idx)
        return ids, oovs

    def encode_target(self, tokens: Sequence[str], oovs: Sequence[str]) -> list[int]:
        """Extended target ids: copyable OOVs get their source slot, others <unk>."""
        ids = []
        for token in tokens:
            idx = self.stoi.get(token)
            if idx is None:
                idx = len(self) + oovs.index(token) if token in oovs else self.unk_id
            ids.append(idx)
        return ids

    def decode(self, ids: Iterable[int], oovs: Sequence[str] = ()) -> list[str]:
        out = []
        for idx in ids:
            if idx < len(self):
                out.append(self.itos[idx])
            elif idx - len(self) < len(oovs):
                out.append(oovs[idx - len(self)])
            else:
                out.append(UNK)
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(itos[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"{path} is not a vocabulary file (special tokens missing)")
        return cls(itos[len(SPECIAL_TOKENS) :])
